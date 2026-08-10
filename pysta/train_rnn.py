
import numpy as np
import time
import pickle
import copy
import os
import sys
import torch
import pysta


def _make_rnn(env, kwargs):
    """Construct the requested recurrent model around an environment."""
    if kwargs["model_type"] == "vanilla":
        return pysta.agents.VanillaRNN(env, **kwargs)
    if kwargs["model_type"] == "lineembedded":
        return pysta.agents.LineEmbeddedRNN(env, **kwargs)
    if kwargs["model_type"] == "corticallyembedded":
        return pysta.agents.CorticallyEmbeddedRNN(env, **kwargs)
    raise ValueError(f"Unknown model_type: {kwargs['model_type']}")


def _stored_execution_accuracy(store):
    """Calculate accuracy only where the environment requests policy loss."""
    corrects = []
    masks = []
    for state in store:
        mask = state.get("policy_loss_mask", state.get("loss_mask"))
        if mask is None:
            # Backward-compatible fallback for stores made by Jensen's agent.
            if state.get("step_num", -1) < 0:
                continue
            mask = ~state["finished"]
        corrects.append(state["corrects"])
        masks.append(mask.to(dtype=torch.bool))
    if not corrects:
        return np.nan
    corrects = torch.stack(corrects)
    masks = torch.stack(masks).to(device=corrects.device)
    corrects = corrects.clone()
    corrects[~masks] = torch.nan
    return float(torch.nanmean(corrects, axis=0).mean().detach().cpu())


def _metric_to_cpu(value):
    """Convert environment metrics to pickle-friendly CPU values."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.numpy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _aggregate_environment_metrics(block_metrics):
    """Combine scalar and per-trial metrics across evaluation blocks."""
    if not block_metrics:
        return {}

    combined = {}
    keys = set().union(*(metrics.keys() for metrics in block_metrics))
    for key in sorted(keys):
        values = [metrics[key] for metrics in block_metrics if key in metrics]
        arrays = [np.asarray(value) for value in values]
        if all(array.ndim == 0 for array in arrays):
            try:
                combined[key] = float(np.mean(arrays))
            except (TypeError, ValueError):
                combined[key] = values
        else:
            try:
                combined[key] = np.concatenate(
                    [array.reshape(1) if array.ndim == 0 else array for array in arrays],
                    axis=0,
                )
            except (TypeError, ValueError):
                combined[key] = values
    return combined


def _aggregate_route_consistency(block_summaries):
    """Pool count-based route metrics without comparing routes across trials."""
    if not block_summaries:
        return {}
    overalls = [summary["overall"] for summary in block_summaries]

    def summed(name):
        return sum(float(overall[name]) for overall in overalls)

    action_pairs = int(summed("action_pair_count"))
    action_matches = int(summed("exact_action_pair_matches"))
    location_pairs = int(summed("location_pair_count"))
    location_matches = int(summed("exact_location_pair_matches"))
    completed_routes = int(summed("completed_route_count"))
    shortest_routes = int(summed("shortest_route_count"))
    navigation_actions = int(summed("navigation_action_count"))
    invalid_actions = int(summed("invalid_boundary_action_count"))
    consistency_routes = int(summed("consistency_route_count"))
    action_modal_routes = sum(
        (overall["action_modal_fraction"] or 0.0)
        * overall["consistency_route_count"]
        for overall in overalls
    )
    location_modal_routes = sum(
        (overall["location_modal_fraction"] or 0.0)
        * overall["consistency_route_count"]
        for overall in overalls
    )
    return {
        "action_source": "env_action",
        "initial_block_start_leg_excluded_from_repetition": True,
        "num_blocks": len(block_summaries),
        "num_trials": sum(summary["num_trials"] for summary in block_summaries),
        "complete_trial_count": sum(
            trial["status"] == "complete"
            for summary in block_summaries
            for trial in summary["trials"]
        ),
        "truncated_trial_count": sum(
            trial["status"] == "truncated"
            for summary in block_summaries
            for trial in summary["trials"]
        ),
        "consistency_route_count": consistency_routes,
        "action_pair_count": action_pairs,
        "exact_action_pair_matches": action_matches,
        "exact_action_pairwise_rate": (
            action_matches / action_pairs if action_pairs else None
        ),
        "location_pair_count": location_pairs,
        "exact_location_pair_matches": location_matches,
        "exact_location_pairwise_rate": (
            location_matches / location_pairs if location_pairs else None
        ),
        "action_modal_fraction": (
            action_modal_routes / consistency_routes
            if consistency_routes
            else None
        ),
        "location_modal_fraction": (
            location_modal_routes / consistency_routes
            if consistency_routes
            else None
        ),
        "completed_route_count": completed_routes,
        "shortest_route_count": shortest_routes,
        "shortest_route_fraction": (
            shortest_routes / completed_routes if completed_routes else None
        ),
        "total_excess_actions": int(summed("total_excess_actions")),
        "mean_excess_actions": (
            summed("total_excess_actions") / completed_routes
            if completed_routes
            else None
        ),
        "navigation_action_count": navigation_actions,
        "invalid_boundary_action_count": invalid_actions,
        "invalid_boundary_action_fraction": (
            invalid_actions / navigation_actions if navigation_actions else None
        ),
    }


@torch.no_grad()
def _evaluate_abcd(rnn, eval_env, num_eval, evaluation_mode):
    """Evaluate complete ABCD blocks on the selected configuration set.

    Each call to ``forward`` resets the evaluation environment and runs one
    complete block. The model is evaluated greedily and on-policy, then its
    training environment and action-selection flags are restored even if an
    evaluation block fails.
    """
    train_env = rnn.env
    train_greedy = rnn.greedy
    train_force_optimal = rnn.force_optimal

    if eval_env.obs_dim != train_env.obs_dim:
        raise ValueError(
            "Training and evaluation environments must have the same "
            f"observation dimension ({train_env.obs_dim} != {eval_env.obs_dim})."
        )
    if eval_env.output_dim != train_env.output_dim:
        raise ValueError(
            "Training and evaluation environments must have the same "
            f"output dimension ({train_env.output_dim} != {eval_env.output_dim})."
        )

    block_losses = []
    block_accuracies = []
    environment_blocks = []
    route_consistency_blocks = []
    try:
        rnn.env = eval_env
        rnn.greedy = True
        rnn.force_optimal = False
        for _ in range(num_eval):
            # BaseAgent.forward calls env.reset() once and retains the full
            # block state sequence. This helper is no-grad, while training
            # remains full-block BPTT.
            block_loss = rnn.forward(store=True)
            block_losses.append(float(block_loss.detach().cpu()))
            block_accuracies.append(_stored_execution_accuracy(rnn.store))
            route_consistency_blocks.append(
                pysta.abcd_analysis_utils.summarize_route_consistency(rnn)
            )
            if hasattr(eval_env, "evaluation_metrics"):
                environment_blocks.append({
                    key: _metric_to_cpu(value)
                    for key, value in eval_env.evaluation_metrics().items()
                })
    finally:
        rnn.env = train_env
        rnn.greedy = train_greedy
        rnn.force_optimal = train_force_optimal

    metrics = {
        "evaluation_mode": str(evaluation_mode),
        "configuration_bank_name": eval_env.bank_name,
        "configuration_bank": tuple(eval_env.configuration_bank),
        "configuration_bank_statistics": (
            pysta.abcd_env.configuration_bank_statistics(
                eval_env.configuration_bank
            )
        ),
        "evaluation_task_seed": int(eval_env.seed),
        "loss": float(np.mean(block_losses)),
        "accuracy": float(np.nanmean(block_accuracies)),
        "block_losses": block_losses,
        "block_accuracies": block_accuracies,
        "num_blocks": int(num_eval),
        "environment": _aggregate_environment_metrics(environment_blocks),
        "environment_blocks": environment_blocks,
        "route_consistency": _aggregate_route_consistency(
            route_consistency_blocks
        ),
        "route_consistency_blocks": route_consistency_blocks,
        "evaluation_recurrent_noise": float(rnn.rec_noise),
    }
    return metrics["loss"], metrics["accuracy"], metrics


def _evaluate_heldout(rnn, eval_env, num_eval):
    """Backward-compatible wrapper for the former held-out-only helper."""
    return _evaluate_abcd(
        rnn,
        eval_env,
        num_eval,
        evaluation_mode="heldout",
    )


def main_train(kwargs):
    
    # print arguments
    print("\n\nSetting up model for training:")
    print(kwargs)

    # set some parameters
    np.random.seed(kwargs["seed"])
    torch.manual_seed(kwargs["seed"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # instantiate model
    task = kwargs.get("task", "maze")
    evaluation_mode = (
        pysta.tasks._abcd_evaluation_mode(kwargs)
        if task == "abcd_fmri"
        else None
    )
    env = pysta.tasks.make_environment(kwargs, split="train")
    eval_env = (
        pysta.tasks.make_environment(kwargs, split="eval")
        if task == "abcd_fmri"
        else None
    )

    # choose between baseline, line-embedded, and cortical-embedded RNN
    rnn = _make_rnn(env, kwargs).to(device)

    # create some filenames and directories
    model_namespace = f"{env.name}/{rnn.name}"
    if task == "abcd_fmri":
        # Best-checkpoint selection depends on the evaluation configuration
        # set, so keep familiar- and held-out-selected runs unambiguous on disk.
        model_namespace = f"{model_namespace}/evaluation_{evaluation_mode}"
    dirname = f"{pysta.utils.basedir}/models/{model_namespace}"
    savename = f"{dirname}/{kwargs['prefix']}model{kwargs['seed']}"
    print("Saving to:")
    print(savename)

    if os.path.isfile(f"{savename}.p"): # if this model already exists, check whether we can overwrite
        if bool(kwargs["overwrite"]):
            print(f"{savename} aleady exists, overwriting!")
        else:
            print(f"{savename} already exists, exiting!")
            raise FileExistsError
        
    if kwargs["save_results"]:
        os.makedirs(f"{dirname}", exist_ok = True)

    # instantiate optimizer and some variables to keep track of
    optim = torch.optim.Adam(rnn.parameters(), lr=kwargs["lrate"])
    all_losses, all_accs, eval_metrics = [], [], []
    best_loss = np.inf
    epoch = -1  # permits a setup/save-only run with num_epochs=0

    # print training message
    time.sleep(5e-2)
    print(f"Training {kwargs['num_epochs']} batches of size {rnn.env.batch} on {device}")

    # now run actual training loop
    t0 = time.time()
    for epoch in range(kwargs["num_epochs"]):
        
        if epoch % kwargs["eval_freq"] == 0:
            with torch.no_grad():
                if task == "abcd_fmri":
                    loss, acc, metrics = _evaluate_abcd(
                        rnn,
                        eval_env,
                        kwargs["num_eval"],
                        evaluation_mode=evaluation_mode,
                    )
                    metrics["epoch"] = epoch
                    eval_metrics.append(metrics)
                else:
                    # Preserve Jensen's original same-environment evaluation.
                    loss, acc = rnn.eval(num_eval = kwargs["num_eval"])
                all_losses.append(loss)
                all_accs.append(acc)
                if loss <= best_loss:
                    best_loss = loss
                    if kwargs["save_results"]:
                        torch.save(rnn, f"{savename}_best.pt")
                
                losses = [np.round(l.item(), 4) for l in [rnn.acc_loss, rnn.ent_loss, rnn.weight_loss, rnn.rate_loss]]
                print(epoch, loss, acc, np.round((time.time() - t0)/60, 2), best_loss, losses)
                sys.stdout.flush()
                
                if kwargs["save_results"]:
                    pickle.dump({"epoch": epoch, "loss": all_losses, "accs": all_accs, "eval_metrics": eval_metrics, "evaluation_mode": evaluation_mode if task == "abcd_fmri" else None, "rnn": rnn, "best_loss": best_loss, "kwargs": kwargs, "optim": optim}, open(f"{savename}.p", "wb"))
                
        optim.zero_grad() # reset gradient accumulator
        loss = rnn.forward() # compute loss
        loss.backward() # compute gradients
        optim.step() # update parameters
        
    if kwargs["save_results"]:
        pickle.dump({"epoch": epoch, "loss": all_losses, "accs": all_accs, "eval_metrics": eval_metrics, "evaluation_mode": evaluation_mode if task == "abcd_fmri" else None, "rnn": rnn, "best_loss": best_loss, "kwargs": kwargs, "optim": optim}, open(f"{savename}.p", "wb"))
        torch.save(rnn, f"{savename}_final.pt")

    return rnn

if __name__ == "__main__": 
    kwargs = pysta.argparser.parse_args()
    main_train(kwargs)
