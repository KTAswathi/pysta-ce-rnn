
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
    """Combine scalar and per-trial metrics across held-out blocks."""
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


def _evaluate_heldout(rnn, eval_env, num_eval):
    """Evaluate complete ABCD blocks on the held-out environment.

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
    try:
        rnn.env = eval_env
        rnn.greedy = True
        rnn.force_optimal = False
        for _ in range(num_eval):
            # BaseAgent.forward calls env.reset() once and retains the full
            # block graph. Evaluation itself runs without gradients in the
            # caller, while training remains full-block BPTT.
            block_loss = rnn.forward(store=True)
            block_losses.append(float(block_loss.detach().cpu()))
            block_accuracies.append(_stored_execution_accuracy(rnn.store))
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
        "loss": float(np.mean(block_losses)),
        "accuracy": float(np.nanmean(block_accuracies)),
        "block_losses": block_losses,
        "block_accuracies": block_accuracies,
        "num_blocks": int(num_eval),
        "environment": _aggregate_environment_metrics(environment_blocks),
        "environment_blocks": environment_blocks,
    }
    return metrics["loss"], metrics["accuracy"], metrics


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
    env = pysta.tasks.make_environment(kwargs, split="train")
    eval_env = (
        pysta.tasks.make_environment(kwargs, split="eval")
        if task == "abcd_fmri"
        else None
    )

    # choose between baseline, line-embedded, and cortical-embedded RNN
    rnn = _make_rnn(env, kwargs).to(device)

    # create some filenames and directories
    dirname = f"{pysta.utils.basedir}/models/{env.name}/{rnn.name}"
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
                    loss, acc, metrics = _evaluate_heldout(
                        rnn,
                        eval_env,
                        kwargs["num_eval"],
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
                    pickle.dump({"epoch": epoch, "loss": all_losses, "accs": all_accs, "eval_metrics": eval_metrics, "rnn": rnn, "best_loss": best_loss, "kwargs": kwargs, "optim": optim}, open(f"{savename}.p", "wb"))
                
        optim.zero_grad() # reset gradient accumulator
        loss = rnn.forward() # compute loss
        loss.backward() # compute gradients
        optim.step() # update parameters
        
    if kwargs["save_results"]:
        pickle.dump({"epoch": epoch, "loss": all_losses, "accs": all_accs, "eval_metrics": eval_metrics, "rnn": rnn, "best_loss": best_loss, "kwargs": kwargs, "optim": optim}, open(f"{savename}.p", "wb"))
        torch.save(rnn, f"{savename}_final.pt")

    return rnn

if __name__ == "__main__": 
    kwargs = pysta.argparser.parse_args()
    main_train(kwargs)
