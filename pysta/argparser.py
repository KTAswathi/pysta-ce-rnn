import argparse


TASK_DEFAULTS = {
    "maze": {
        "model_type": "corticallyembedded",
        "Nrec": 347,
        "batch_size": 200,
        "embedding_name": "mpfc_union_gradient_nearest_a24_25_anchor25",
        "embedding_species": "human",
        "embedding_seed": 42,
    },
    "abcd_fmri": {
        "model_type": "corticallyembedded",
        "Nrec": 480,
        "batch_size": 8,
        "embedding_name": "mpfc_projected_mask_linear0p1",
        "embedding_species": "human",
        "embedding_seed": 42,
    },
}


def apply_task_defaults(parameters):
    """Return a copy with task-conditioned model/batch defaults filled."""
    parameters = dict(parameters)
    task = parameters.get("task", "maze")
    if task not in TASK_DEFAULTS:
        raise ValueError(f"Unknown task: {task}")
    for key, value in TASK_DEFAULTS[task].items():
        if parameters.get(key) is None:
            parameters[key] = value
    return parameters


def parse_args(**kwargs):
    """
    Parameters
    -----------
    new_kwargs : dict
        Additional keyword arguments to override the default values.
    """
    
    parser = argparse.ArgumentParser()

    # task selection. Keep MazeEnv as the default so existing Jensen commands
    # continue to construct exactly the same task.
    parser.add_argument(
        '--task',
        choices=['maze', 'abcd_fmri'],
        default='maze',
        help="behavioural task to train: Jensen's maze or the ABCD fMRI task",
    )

    # environment args
    parser.add_argument('--side_length', type=int, default=4, help="of arena")
    parser.add_argument('--max_steps', type=int, default=6, help="in a trial")
    parser.add_argument('--changing_trial_maze', default=0, type=int, help="does the maze change between trials")
    parser.add_argument('--dynamic_rew', default=1, type=int, help="does reward function vary in time")
    parser.add_argument('--sample_wall_num', default=10, type=int, help="how many different mazes in each batch")
    parser.add_argument('--rew_landscape', default=1, type=int, help="if true, the reward function is iid in space and time. Otherwise an absorbing goal is used.")
    parser.add_argument('--relative_rew', default=1, type=int, help="is the reward input relative in time?")
    parser.add_argument('--output_format', default="allocentric", type=str, help="egocentric or allocentric")
    parser.add_argument('--planning_steps', nargs="+", type=int, default=[5, 6, 7], help="how many 'planning steps' of the environment before the 'execution period'?")
    parser.add_argument('--working_memory', default=1, type=int, help="if true, no reward input during execution")
    parser.add_argument('--inp_noise', type=float, default=1e-3, help="noise fraction during the execution period")
    parser.add_argument('--inp_noise_planning', type=float, default=1e-3, help="noise fraction during the planning period")

    # model args
    # These task-dependent defaults are filled after command-line and
    # programmatic overrides have both been applied (see below).
    parser.add_argument('--Nrec', type=int, default=None, help="number of hidden units")
    parser.add_argument('--nonlin_output', default=0, type=int, help="if true, include a hidden layer in the output function from the RNN")
    parser.add_argument('--r_reg', type=float, default=1e-5, help="rate regularization strength")
    parser.add_argument('--W_reg', type=float, default=2e-7, help="weight regularization strength")
    parser.add_argument('--ent_reg', type=float, default=1e-4, help="entropy regularization strength")
    parser.add_argument('--rec_noise', type=float, default=1e-3, help="recurrent noise magnitude")
    parser.add_argument('--force_optimal', default=1, type=int, help="if true, force the agent to follow an optimal trajectory during training")
    parser.add_argument('--iters_per_action', nargs="+", type=int, default=[10], help="number of RNN iterations per environment step")
    parser.add_argument('--tau', default=5.0, type=float, help="RNN update timescale. 1 imposes no external autocorrelation.")

    # mini-CERNN args
    parser.add_argument(
        '--model_type',
        default=None,
        type=str,
        help="vanilla, lineembedded, or corticallyembedded",
    )
    parser.add_argument('--dist_reg', type=float, default=1e-7, help="distance-weighted recurrent regularisation strength")
    parser.add_argument('--use_local_init', default=1, type=int, help="if true, use distance-biased recurrent init")
    parser.add_argument('--line_decay', type=float, default=0.12, help="distance decay for the geometry-based init / regulariser")
    parser.add_argument('--line_init_scale', type=float, default=1.0, help="scale of locality-biased recurrent init")

    # local input/output anatomy options
    parser.add_argument('--localize_loc_input', default=1, type=int, help="if true, current-location input is routed to one-end local band")
    parser.add_argument('--localize_rew_input', default=0, type=int, help="if true, reward input is routed to one-end local band")
    parser.add_argument('--localize_wall_input', default=0, type=int, help="if true, wall input is routed to one-end local band")
    parser.add_argument('--local_fraction', type=float, default=1.0 / 6.0, help="fraction of recurrent units in the local band")
    parser.add_argument('--readout_mode', default='global', type=str, help="global, same_end, or opposite_end")

    # cortical embedding options
    parser.add_argument(
        '--embedding_name',
        default=None,
        type=str,
        help="name of saved cortical embedding roi",
    )
    parser.add_argument('--embedding_species', default=None, type=str, help="species folder for saved embedding")
    parser.add_argument('--embedding_seed', default=None, type=int, help="seed used when generating the saved embedding")
    parser.add_argument(
        '--anchor_area_names',
        nargs="+",
        default=["25"],
        help="sampled parcel labels used to define the proxy hippocampal-facing anchor",
    )
    parser.add_argument(
        '--anchor_vertex_file',
        default=None,
        type=str,
        help=(
            "Optional .npy file defining explicit anchor/input vertices. "
            "If provided, this overrides anchor_area_names for CorticallyEmbeddedRNN. "
            "Can be an absolute path, a path relative to repo root, or a filename in "
            "data/embedding/custom_roi_vertices."
        ),
    )

    # training args
    parser.add_argument('--batch_size', type=int, default=None, help="batch size (task-dependent default: maze=200, abcd_fmri=8)")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--overwrite', default=0, type=int, help="allow overwrite of existing model of the same name")
    parser.add_argument('--eval_freq', type=int, default=200, help="number of batches between each instance of evaluation and model saving")
    parser.add_argument('--num_eval', type=int, default=10, help="number of batches to use for evaluation")
    parser.add_argument('--num_epochs', type=int, default=200000, help="number of epochs to train for")
    parser.add_argument('--prefix', type=str, default="", help="optional prefix to the model name")
    parser.add_argument('--lrate', type=float, default=3e-4, help="ADAM learning rate")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save the model")

    # ABCD fMRI-task arguments. They are harmless extras for MazeEnv, whose
    # constructor already accepts unused keyword arguments. Configuration
    # strings encode ordered location IDs separated by semicolons; parsing the
    # task-level structure is delegated to pysta.tasks / pysta.abcd_env.
    parser.add_argument('--n_loops', type=int, default=None, help="number of instruction/execution loops in an ABCD block")
    parser.add_argument('--instruction_repeats', type=int, default=None, help="number of presentations of each ABCD instruction")
    parser.add_argument('--configuration_seed', type=int, default=0, help="base seed used when task-specific configuration seeds are omitted")
    parser.add_argument('--train_configuration_seed', type=int, default=None, help="seed for generating the ABCD training configuration bank")
    parser.add_argument('--eval_configuration_seed', type=int, default=None, help="seed for generating the held-out ABCD evaluation configuration bank")
    parser.add_argument('--train_configurations', type=str, default=None, help="explicit semicolon-separated ordered ABCD training configurations")
    parser.add_argument('--eval_configurations', type=str, default=None, help="explicit semicolon-separated ordered ABCD evaluation configurations")
    parser.add_argument('--num_train_configurations', type=int, default=None, help="number of generated ABCD training configurations")
    parser.add_argument('--num_eval_configurations', type=int, default=None, help="number of generated held-out ABCD evaluation configurations")
    parser.add_argument('--train_task_seed', type=int, default=None, help="RNG seed for sampling ABCD training blocks")
    parser.add_argument('--eval_task_seed', type=int, default=None, help="RNG seed for sampling held-out ABCD evaluation blocks")
    parser.add_argument(
        '--start_position_policy',
        choices=['exclude_first_goal', 'uniform', 'fixed'],
        default=None,
        help="ABCD non-target start policy; fixed also requires --start_position",
    )
    parser.add_argument('--start_position', type=int, default=None, help="fixed ABCD start location ID")
    parser.add_argument('--max_navigation_steps', type=int, default=None, help="maximum ABCD navigation actions in one block")
    parser.add_argument('--instruction_directions', nargs='+', type=str, default=None, help="allowed ABCD instruction directions")
    parser.add_argument('--execution_relations', nargs='+', type=str, default=None, help="allowed ABCD execution relations")
    parser.add_argument('--min_goal_distance', type=int, default=None, help="minimum circular consecutive-goal Manhattan distance")

    # parse command line arguments
    parameters = vars(parser.parse_args())

    for key, value in kwargs.items():
        parameters[key] = value

    parameters = apply_task_defaults(parameters)
        
    bool_parameters = [
        "changing_trial_maze",
        "dynamic_rew",
        "rew_landscape",
        "relative_rew",
        "working_memory",
        "nonlin_output",
        "force_optimal",
        "save_results",
        "use_local_init",
        "localize_loc_input",
        "localize_rew_input",
        "localize_wall_input",
    ]

    for parameter in bool_parameters:
        parameters[parameter] = bool(parameters[parameter])
    
    if (type(parameters["iters_per_action"]) != int) and (len(parameters["iters_per_action"]) == 1):
        parameters["iters_per_action"] = parameters["iters_per_action"][0]
        
    return parameters
