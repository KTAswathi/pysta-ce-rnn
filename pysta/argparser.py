import argparse

def parse_args(**kwargs):
    """
    Parameters
    -----------
    new_kwargs : dict
        Additional keyword arguments to override the default values.
    """
    
    parser = argparse.ArgumentParser()

    # environment args
    parser.add_argument('--side_length', type = int, default = 4, help = "of arena")
    parser.add_argument('--max_steps', type = int, default = 6, help = "in a trial")
    parser.add_argument('--changing_trial_maze', default = 0, type = int, help = "does the maze change between trials")
    parser.add_argument('--dynamic_rew', default = 1, type = int, help = "does reward function vary in time")
    parser.add_argument('--sample_wall_num', default = 10, type = int, help = "how many different mazes in each batch")
    parser.add_argument('--rew_landscape', default = 1, type = int, help = "if true, the reward function is iid in space and time. Otherwise an absorbing goal is used.")
    parser.add_argument('--relative_rew', default = 1, type = int, help = "is the reward input relative in time?")
    parser.add_argument('--output_format', default = "allocentric", type = str, help = "egocentric or allocentric")
    parser.add_argument('--planning_steps', nargs="+", type = int, default = [5,6,7], help = "how many 'planning steps' of the environment before the 'execution period'?")
    parser.add_argument('--working_memory', default = 1, type = int, help = "if true, no reward input during execution")
    parser.add_argument('--inp_noise', type = float, default = 1e-3, help = "noise fraction during the execution period")
    parser.add_argument('--inp_noise_planning', type = float, default = 1e-3, help = "noise fraction during the planning period")

    # model args
    parser.add_argument('--Nrec', type = int, default = 800, help = "number of hidden units")
    parser.add_argument('--nonlin_output', default = 0, type = int, help = "if true, include a hidden layer in the output function from the RNN")
    parser.add_argument('--r_reg', type = float, default = 1e-5, help = "rate regularization strength")
    parser.add_argument('--W_reg', type = float, default = 2e-7, help = "weight regularization strength")
    parser.add_argument('--ent_reg', type = float, default = 1e-4, help = "entropy regularization strength")
    parser.add_argument('--rec_noise', type = float, default = 1e-3, help = "recurrent noise magnitude")
    parser.add_argument('--force_optimal', default = 1, type = int, help = "if true, force the agent to follow an optimal trajectory during training")
    parser.add_argument('--iters_per_action', nargs="+", type = int, default = [10], help = "number of RNN iterations per environment step")
    parser.add_argument('--tau', default = 5.0, type = float, help = "RNN update timescale. 1 imposes no external autocorrelation.")

    # training args
    parser.add_argument('--batch_size', type = int, default = 200, help = "batch size for the environment")
    parser.add_argument('--seed', type = int, default = 0, help = "random seed")
    parser.add_argument('--overwrite', default = 0, type = int, help = "allow overwrite of existing model of the same name")
    parser.add_argument('--eval_freq', type = int, default = 200, help = "number of batches between each instance of evaluation and model saving")
    parser.add_argument('--num_eval', type = int, default = 10, help = "number of batches to use for evaluation")
    parser.add_argument('--num_epochs', type = int, default = 200000, help = "number of epochs to train for")
    parser.add_argument('--prefix', type = str, default = "", help = "optional prefix to the model name")
    parser.add_argument('--lrate', type = float, default = 3e-4, help = "ADAM learning rate")
    parser.add_argument('--save_results', type = int, default = 1, help = "whether to save the model")

    # parse command line arguments
    parameters = vars(parser.parse_args())
    for key, value in kwargs.items():
        parameters[key] = value
        
    bool_parameters = ["changing_trial_maze", "dynamic_rew", "rew_landscape", "relative_rew", "working_memory", "nonlin_output", "force_optimal", "save_results"]
    for parameter in bool_parameters:
        parameters[parameter] = bool(parameters[parameter])
    
    if (type(parameters["iters_per_action"]) != int) and (len(parameters["iters_per_action"]) == 1):
        parameters["iters_per_action"] = parameters["iters_per_action"][0] # just the single possibility
        
    return parameters
