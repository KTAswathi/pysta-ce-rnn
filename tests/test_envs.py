"""Code for testing that the maze environment runs and behaves as expected"""

#%%
import pysta
import torch
import numpy as np

def test_maze_env_runs():
    """
    test that environments can be instantiated, generated optimal actions, and update from those actions
    """
    pysta.reload()
    # try a bunch of different combinations of things
    print("\nInstantiating different environments")
    for relative_rew in [True, False]:
        for output_format in ["egocentric", "allocentric"]:
            for planning_steps in [0, 3, [3,4,5]]:
                for working_memory in [True, False]:
                    for dynamic_rew in [True, False]:
                        for changing_trial_rew in [True, False]:
                            for rew_landscape in [True, False]:

                                env = pysta.envs.MazeEnv(relative_rew = relative_rew, output_format = output_format, planning_steps = planning_steps, working_memory = working_memory, changing_trial_rew = changing_trial_rew, rew_landscape = rew_landscape)
                                obs = env.observation() # generate observation
                                optimal_actions = env.optimal_actions() # compute optimal actions
                                actions = torch.multinomial(optimal_actions, 1)[:, 0] # sample actual action
                                env.step(actions) # step through the environment
                                    
    for side_length in [4,5]:
        for max_steps in [6,7]:
            for batch_size in [1,5]:
                for sample_wall_num in [None, 8]:
                    for inp_noise in [0.0, 1e-1]:
                        for inp_noise_planning in [None, 1e-1]:
                            
                            env = pysta.envs.MazeEnv(side_length = side_length, max_steps = max_steps, inp_noise = inp_noise, inp_noise_planning = inp_noise_planning, batch_size = batch_size, sample_wall_num = sample_wall_num, )
                            obs = env.observation() # generate observation
                            optimal_actions = env.optimal_actions() # compute optimal actions
                            actions = torch.multinomial(optimal_actions, 1)[:, 0] # sample actual action
                            env.step(actions) # step through the environment
    
    print("Done!")
    return

def test_cumulative_value():
    
    print("\nTesting value calculations")
    
    pysta.reload()
    
    # compute empirical reward following optimal actions (just for rew landscape since the other tasks have early stopping which is slightly different)
    env = pysta.envs.MazeEnv(rew_landscape = True, batch_size = 11)
                
    loc0s = env.loc
    while not torch.all(env.finished): # as long as there are some trials left to act in
        optimal_actions = env.optimal_actions()
        acts = torch.multinomial(optimal_actions, 1)[:, 0]
        env.step(acts) # action passed to the environment (optionally restricted to be optimal)
    
    cum_rews = env.sample_rews.sum(-1)
    v0s = env.vs[env.batch_inds, 0, loc0s]
    
    # check that empirical reward matches value when following optimal policy
    assert torch.all(torch.isclose(cum_rews, v0s))
    
    print("Done!")
    return

#%%
print("\n\nTesting envs!")

#%%
test_maze_env_runs()

#%%
test_cumulative_value()



# %%
