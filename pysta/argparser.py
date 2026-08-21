import argparse


# Keep command-line applicability visible and machine-searchable.  In
# particular, ``[ABCD cortical]`` means that the option affects an ABCD run
# when the cortically embedded model is selected; it does not imply that the
# option is meaningful for the vanilla model.
ABCD_AND_MAZE = "[ABCD + maze]"
ABCD_TASK = "[ABCD task]"
ABCD_EMBEDDED = "[ABCD cortical/line-embedded]"
ABCD_CORTICAL = "[ABCD cortical]"
MAZE_ONLY = "[maze only; ignored by ABCD]"
ABCD_ROUTING_IGNORED = "[ABCD: routing ignored]"
ABCD_TRAINING_IGNORED = "[ABCD: training ignored]"


def _tagged(tag, description):
    """Prefix an argparse description with a consistent applicability tag."""
    return f"{tag} {description}"


TASK_DEFAULTS = {
    "maze": {
        "model_type": "corticallyembedded",
        "Nrec": 347,
        "batch_size": 200,
        "embedding_name": "mpfc_union_gradient_nearest_a24_25_anchor25",
        "embedding_species": "human",
        "embedding_seed": 42,
        "local_fraction": 1.0 / 6.0,
        "lrate": 3e-4,
    },
    "abcd_fmri": {
        "model_type": "corticallyembedded",
        "Nrec": 480,
        "batch_size": 8,
        "embedding_name": "mpfc_projected_mask_linear0p1",
        "embedding_species": "human",
        "embedding_seed": 42,
        "local_fraction": 1.0 / 4.0,
        "lrate": 1e-4,
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

    # task selection. Keep MazeEnv as the default so existing commands continue
    # to construct exactly the same task.
    parser.add_argument(
        '--task',
        choices=['maze', 'abcd_fmri'],
        default='maze',
        help=_tagged(ABCD_AND_MAZE, "behavioural task to train: maze or ABCD fMRI"),
    )

    # environment args
    parser.add_argument('--side_length', type=int, default=4, help=_tagged(MAZE_ONLY, "arena side length; ABCD is fixed at 3"))
    parser.add_argument('--max_steps', type=int, default=6, help=_tagged(MAZE_ONLY, "movement steps in a trial"))
    parser.add_argument('--changing_trial_maze', default=0, type=int, help=_tagged(MAZE_ONLY, "whether the maze changes between trials"))
    parser.add_argument('--dynamic_rew', default=1, type=int, help=_tagged(MAZE_ONLY, "whether the reward function varies in time"))
    parser.add_argument('--sample_wall_num', default=10, type=int, help=_tagged(MAZE_ONLY, "number of different mazes in each batch"))
    parser.add_argument('--rew_landscape', default=1, type=int, help=_tagged(MAZE_ONLY, "if true, use an iid spatial/temporal reward function; otherwise use an absorbing goal"))
    parser.add_argument('--relative_rew', default=1, type=int, help=_tagged(MAZE_ONLY, "whether the reward input is relative in time"))
    parser.add_argument('--output_format', default="allocentric", type=str, help=_tagged(MAZE_ONLY, "egocentric or allocentric output"))
    parser.add_argument('--planning_steps', nargs="+", type=int, default=[5, 6, 7], help=_tagged(MAZE_ONLY, "environment planning steps before execution; ABCD has no planning period"))
    parser.add_argument('--working_memory', default=1, type=int, help=_tagged(MAZE_ONLY, "if true, suppress reward input during execution"))
    parser.add_argument('--inp_noise', type=float, default=1e-3, help=_tagged(MAZE_ONLY, "input-noise fraction during execution"))
    parser.add_argument('--inp_noise_planning', type=float, default=1e-3, help=_tagged(MAZE_ONLY, "input-noise fraction during planning"))

    # model args
    # These task-dependent defaults are filled after command-line and
    # programmatic overrides have both been applied (see below).
    parser.add_argument('--Nrec', type=int, default=None, help=_tagged(ABCD_AND_MAZE, "number of recurrent hidden units (task-conditioned default)"))
    parser.add_argument('--nonlin_output', default=0, type=int, help=_tagged(ABCD_AND_MAZE, "if true, include a hidden layer in the RNN readout"))
    parser.add_argument('--r_reg', type=float, default=1e-5, help=_tagged(ABCD_AND_MAZE, "firing-rate regularisation strength"))
    parser.add_argument('--W_reg', type=float, default=2e-7, help=_tagged(ABCD_AND_MAZE, "L2 parameter-regularisation strength"))
    parser.add_argument('--ent_reg', type=float, default=1e-4, help=_tagged(ABCD_AND_MAZE, "policy-entropy regularisation strength"))
    parser.add_argument('--rec_noise', type=float, default=1e-3, help=_tagged(ABCD_AND_MAZE, "recurrent-noise magnitude"))
    parser.add_argument('--force_optimal', default=1, type=int, help=_tagged(ABCD_AND_MAZE, "if true, use an optimal environment action to advance training trajectories; ABCD evaluation remains autonomous"))
    parser.add_argument('--iters_per_action', nargs="+", type=int, default=[10], help=_tagged(ABCD_AND_MAZE, "recurrent microsteps per environment timestep"))
    parser.add_argument('--tau', default=5.0, type=float, help=_tagged(ABCD_AND_MAZE, "leaky-RNN update timescale; 1 removes externally imposed autocorrelation"))

    # mini-CERNN args
    parser.add_argument(
        '--model_type',
        default=None,
        type=str,
        help=_tagged(ABCD_AND_MAZE, "RNN type: vanilla, lineembedded, or corticallyembedded (task-conditioned default)"),
    )
    parser.add_argument('--dist_reg', type=float, default=1e-7, help=_tagged(ABCD_EMBEDDED, "distance-weighted recurrent-connection regularisation strength"))
    parser.add_argument('--use_local_init', default=1, type=int, help=_tagged(ABCD_EMBEDDED, "if true, apply distance-biased recurrent initialisation"))
    parser.add_argument('--line_decay', type=float, default=0.12, help=_tagged(ABCD_EMBEDDED, "distance-decay scale for recurrent initialisation"))
    parser.add_argument('--line_init_scale', type=float, default=1.0, help=_tagged(ABCD_EMBEDDED, "scale of recurrent weights before locality modulation"))

    # local input/output anatomy options
    parser.add_argument('--localize_loc_input', default=1, type=int, help=_tagged(ABCD_ROUTING_IGNORED, "legacy maze-routing flag; it is stored in the model name but ABCD routes current and instruction locations locally through its environment interface"))
    parser.add_argument('--localize_rew_input', default=0, type=int, help=_tagged(ABCD_ROUTING_IGNORED, "legacy maze-routing flag; it is stored in the model name but ABCD routes reward/context inputs globally through its environment interface"))
    parser.add_argument('--localize_wall_input', default=0, type=int, help=_tagged(ABCD_ROUTING_IGNORED, "legacy maze-routing flag; it is stored in the model name but ABCD has no wall-input group"))
    parser.add_argument('--local_fraction', type=float, default=None, help=_tagged(ABCD_EMBEDDED, "fraction of recurrent units in each local input/readout band (task default: maze=1/6, ABCD=1/4)"))
    parser.add_argument('--readout_mode', default='global', type=str, help=_tagged(ABCD_EMBEDDED, "readout routing: global, same_end, or opposite_end"))

    # cortical embedding options
    parser.add_argument(
        '--embedding_name',
        default=None,
        type=str,
        help=_tagged(ABCD_CORTICAL, "name of the saved cortical-embedding ROI (task-conditioned default)"),
    )
    parser.add_argument('--embedding_species', default=None, type=str, help=_tagged(ABCD_CORTICAL, "species folder containing the saved embedding (task-conditioned default)"))
    parser.add_argument('--embedding_seed', default=None, type=int, help=_tagged(ABCD_CORTICAL, "seed identifying the saved cortical embedding (task-conditioned default)"))
    parser.add_argument(
        '--anchor_area_names',
        nargs="+",
        default=["25"],
        help=_tagged(ABCD_CORTICAL, "fallback parcel labels defining the proxy hippocampal-facing anchor when the embedding has no anchor_unit_indices.npy"),
    )
    parser.add_argument(
        '--anchor_vertex_file',
        default=None,
        type=str,
        help=(
            f"{ABCD_TRAINING_IGNORED} embedding-generation input only; the training "
            "constructor does not consume this option. Create the saved "
            "embedding/anchor files before training instead."
        ),
    )

    # training args
    parser.add_argument('--batch_size', type=int, default=None, help=_tagged(ABCD_AND_MAZE, "training batch size (task default: maze=200, ABCD=8)"))
    parser.add_argument('--seed', type=int, default=0, help=_tagged(ABCD_AND_MAZE, "model/training random seed; also the default ABCD training-task seed"))
    parser.add_argument('--overwrite', default=0, type=int, help=_tagged(ABCD_AND_MAZE, "allow overwriting an existing legacy-layout model; managed --run_name directories never overwrite"))
    parser.add_argument('--eval_freq', type=int, default=200, help=_tagged(ABCD_AND_MAZE, "training batches between evaluation/checkpoint operations"))
    parser.add_argument('--num_eval', type=int, default=10, help=_tagged(ABCD_AND_MAZE, "evaluation batches/blocks per monitoring operation"))
    parser.add_argument(
        '--evaluation_mode',
        choices=['familiar', 'heldout'],
        default='familiar',
        help=(
            f"{ABCD_TASK} evaluation configuration set: 'familiar' reuses the "
            "training/familiarisation bank for ordinary performance monitoring, "
            "whereas 'heldout' evaluates schema generalisation on a disjoint bank"
        ),
    )
    parser.add_argument('--num_epochs', type=int, default=200000, help=_tagged(ABCD_AND_MAZE, "number of optimizer steps/training batches"))
    parser.add_argument('--prefix', type=str, default="", help=_tagged(ABCD_AND_MAZE, "optional prefix for the saved model name"))
    parser.add_argument('--lrate', type=float, default=None, help=_tagged(ABCD_AND_MAZE, "Adam learning rate (task default: maze=3e-4, ABCD=1e-4)"))
    parser.add_argument('--save_results', type=int, default=1, help=_tagged(ABCD_AND_MAZE, "whether to save checkpoints and training metadata"))
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help=_tagged(
            ABCD_TASK,
            "short human-readable name enabling the managed models/abcd_fmri/<name>_<config-hash>/ layout; omitted preserves the legacy layout",
        ),
    )
    parser.add_argument(
        '--resume',
        default=0,
        type=int,
        help=_tagged(
            ABCD_TASK,
            "resume an interrupted managed --run_name launch from checkpoints/latest.pt; the exact resolved configuration must match",
        ),
    )

    # ABCD fMRI-task arguments. They are harmless extras for MazeEnv, whose
    # constructor already accepts unused keyword arguments. Configuration
    # strings encode ordered location IDs separated by semicolons; parsing the
    # task-level structure is delegated to pysta.tasks / pysta.abcd_env.
    parser.add_argument('--n_loops', type=int, default=None, help=_tagged(ABCD_TASK, "continuous four-goal loops in one recurrent block (default 5)"))
    parser.add_argument('--instruction_repeats', type=int, default=None, help=_tagged(ABCD_TASK, "presentations of the four-location instruction (default 2)"))
    parser.add_argument('--configuration_seed', type=int, default=0, help=_tagged(ABCD_TASK, "base seed used when task-specific configuration seeds are omitted"))
    parser.add_argument('--train_configuration_seed', type=int, default=None, help=_tagged(ABCD_TASK, "seed for generating the training configuration bank"))
    parser.add_argument('--eval_configuration_seed', type=int, default=None, help=_tagged(ABCD_TASK, "seed for generating the held-out evaluation bank; used only with --evaluation_mode heldout"))
    parser.add_argument('--train_configurations', type=str, default=None, help=_tagged(ABCD_TASK, "explicit semicolon-separated ordered training configurations"))
    parser.add_argument('--familiar_configurations', type=str, default=None, help=_tagged(ABCD_TASK, "optional exact ordered training-bank subset for familiar monitoring"))
    parser.add_argument('--eval_configurations', type=str, default=None, help=_tagged(ABCD_TASK, "explicit semicolon-separated held-out configurations; used only with --evaluation_mode heldout"))
    parser.add_argument(
        '--synthetic_fmri_bank_objective',
        choices=['balance_first', 'distance_first'],
        default=None,
        help=(
            f"{ABCD_TASK} use a deterministic synthetic "
            "10-configuration/five-inverse-pair "
            "training/comparison bank; explicitly choose the location-balance "
            "versus mean-distance priority (these are not Svenja's coordinates "
            "and this option is not used by final factorial fMRI evaluation)"
        ),
    )
    parser.add_argument('--num_train_configurations', type=int, default=None, help=_tagged(ABCD_TASK, "number of generated training physical-cycle representatives (default 12)"))
    parser.add_argument('--num_eval_configurations', type=int, default=None, help=_tagged(ABCD_TASK, "number of generated held-out physical-cycle representatives (default 6; heldout mode only)"))
    parser.add_argument('--train_task_seed', type=int, default=None, help=_tagged(ABCD_TASK, "RNG seed for sampling training blocks"))
    parser.add_argument('--eval_task_seed', type=int, default=None, help=_tagged(ABCD_TASK, "independent RNG seed for sampling monitoring blocks"))
    parser.add_argument(
        '--start_position_policy',
        choices=['exclude_first_goal', 'uniform', 'fixed'],
        default=None,
        help=_tagged(ABCD_TASK, "start policy: default excludes the first target; uniform samples all 9 cells; fixed also requires --start_position"),
    )
    parser.add_argument('--start_position', type=int, default=None, help=_tagged(ABCD_TASK, "fixed start-location ID used with --start_position_policy fixed"))
    parser.add_argument('--max_navigation_steps', type=int, default=None, help=_tagged(ABCD_TASK, "maximum navigation actions in one block (default 200)"))
    parser.add_argument('--instruction_directions', nargs='+', type=str, default=None, help=_tagged(ABCD_TASK, "allowed instruction directions (FORWARD/BACKWARD)"))
    parser.add_argument('--execution_relations', nargs='+', type=str, default=None, help=_tagged(ABCD_TASK, "allowed execution relations (SAME/REVERSE)"))
    parser.add_argument('--min_goal_distance', type=int, default=None, help=_tagged(ABCD_TASK, "minimum goal-to-goal Manhattan distance; generated/fMRI banks enforce it for every pair"))
    parser.add_argument(
        '--fmri_base_configurations',
        type=str,
        default=None,
        help=(
            f"{ABCD_TASK} optional override containing exactly five familiar "
            "base spatial "
            "configurations for the final scanner-style evaluation; without "
            "it, a documented synthetic balance-controlled five-base fallback "
            "is used; direction/reversal is crossed factorially and must not "
            "be duplicated in this bank"
        ),
    )
    parser.add_argument(
        '--fmri_evaluation_seed',
        type=int,
        default=None,
        help=(
            f"{ABCD_TASK} deterministic task/start and recurrent-noise seed "
            "for the final "
            "20-block factorial fMRI evaluation"
        ),
    )
    parser.add_argument(
        '--run_final_fmri_evaluation',
        default=0,
        type=int,
        help=(
            f"{ABCD_TASK} after training, run once over five resolved base "
            "configurations "
            "x two instruction directions x two execution relations"
        ),
    )

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
        "run_final_fmri_evaluation",
        "resume",
    ]

    for parameter in bool_parameters:
        parameters[parameter] = bool(parameters[parameter])
    
    if (type(parameters["iters_per_action"]) != int) and (len(parameters["iters_per_action"]) == 1):
        parameters["iters_per_action"] = parameters["iters_per_action"][0]
        
    return parameters
