"""Code for testing that all handcrafted agents run and behave as expected"""

#%%

import pysta
import torch
from pysta import basedir

def test_RNN_agent_runs():
    
    pysta.reload()
    print("\nInstantiating a bunch of agents and running a trial")
    env = pysta.envs.MazeEnv(batch = 5)
    
    for force_optimal in [True, False]:
        for nonlin_output in [True, False]:
            for iters_per_action in [1,5]:
                for tau in [1,5,10]:
                    agent = pysta.agents.VanillaRNN(env, Nrec = 20)
                    loss = agent.forward()
        
    print("Done!")
    return
        
        
def test_trial_plot():
    pysta.reload()
    
    print("\nPlotting optimal policies")
    for iout, output_format in enumerate(["egocentric", "allocentric"]):
        for ienv, env in enumerate([pysta.envs.MazeEnv(output_format = output_format),
                                    pysta.envs.MazeEnv(rew_landscape = False, output_format = output_format),
                                    pysta.envs.MazeEnv(rew_landscape = False, dynamic_rew = False, output_format = output_format)]):
            
            rnn = pysta.agents.VanillaRNN(env, force_optimal = True)
            rnn.plot_trial(run_trial = True, filename = f"{basedir}/figures/tests/trial{ienv}{iout}.png")
            rnn.plot_trial(run_trial = False, values = False, filename = f"{basedir}/figures/tests/trial{ienv}{iout}_r.png")

    print("Done!")
    return

def test_STA():
    print("\nTesting STA agent")
    pysta.reload()
    
    for output_format in ["allocentric", "egocentric"]:
        for ienv, env in enumerate([
            pysta.envs.MazeEnv(rew_landscape = False, output_format = output_format, batch_size = 101, dynamic_rew = False),
            pysta.envs.MazeEnv(rew_landscape = False, output_format = output_format, batch_size = 101, dynamic_rew = True),
            pysta.envs.MazeEnv(rew_landscape = True, output_format = output_format, batch_size = 101, dynamic_rew = True)
            ]):
            
            pysta.reload()

            sta = pysta.agents.SpaceTimeAttractor(env, greedy = True)
            
            frac_opt = pysta.utils.optimal_initial_action_freq(sta, reset_after = False)
            sta.plot_representation(filename = f"{basedir}/figures/tests/sta{ienv}_{output_format}_1.png", trial_num = 0)
            print(output_format, ienv, frac_opt)
            assert frac_opt > (0.85 if ienv == 2 else 0.95)

    print("Done!")
    return


def test_SR():
    print("\nTesting SR agent")
    pysta.reload()
    for output_format in ["allocentric", "egocentric"]:
        for ienv, env in enumerate([
            pysta.envs.MazeEnv(rew_landscape = False, output_format = output_format, batch_size = 101, dynamic_rew = False),
            pysta.envs.MazeEnv(rew_landscape = False, output_format = output_format, batch_size = 101, dynamic_rew = True),
            pysta.envs.MazeEnv(rew_landscape = True, output_format = output_format, batch_size = 101, dynamic_rew = True)
            ]):
            
            env.inp_noise_planning, env.inp_noise = 0.0, 0.0
            sr = pysta.agents.SRLearner(env, greedy = True, force_optimal = False)
            
            frac_opt = pysta.utils.optimal_initial_action_freq(sr)
            print(output_format, ienv, frac_opt)
            
            if ienv == 0: # static goal should have perfect performance for SR
                assert frac_opt > 0.99
            
            sr.plot_trial(run_trial = True, filename = f"{basedir}/figures/tests/sr{ienv}_{output_format}_true.png")
            sr.plot_trial_values(run_trial = False, filename = f"{basedir}/figures/tests/sr{ienv}_{output_format}_est.png")
        
    print("Done!")
    return

def test_TD():
    print("\ntesting TD agent")
    pysta.reload()
    
    ntrain = 500
    for ichange, changing_rew in enumerate([False, True]):
        for idyn, dynamic_rew in enumerate([False, True]):
            # test in an environment with a constant reward function
            env = pysta.envs.MazeEnv(rew_landscape = False, output_format = "allocentric", batch_size = 51, dynamic_rew = dynamic_rew, changing_trial_rew = changing_rew)

            td = pysta.agents.TDLearner(env, force_optimal = False, beta = 5.0)

            emp_rews = torch.zeros(ntrain)
            for iter_ in range(ntrain): # train for 500 trials
                td.forward()
                emp_rews[iter_] = env.sample_rews.sum(-1).mean()
            emp1, emp2 = emp_rews[:2].mean(), emp_rews[-2:].mean()
            
            td.plot_trial(run_trial = True, filename = f"{basedir}/figures/tests/td_true_{idyn}_{ichange}.png")
            td.plot_trial_values(run_trial = False, filename = f"{basedir}/figures/tests/td_est_{idyn}_{ichange}.png")
            
            assert td.values.max() <= 1.0
            
            if not (dynamic_rew or changing_rew):
                assert emp2 > emp1
            
            ### count how often initial action is optimal ###

            td.greedy = True
            frac_opt = pysta.utils.optimal_initial_action_freq(td)
            print("Opt:", frac_opt, changing_rew, dynamic_rew)
            
            if not (dynamic_rew or changing_rew):
                assert frac_opt > 0.99


#%%
print("\n\nTesting agents!")

#%%
test_RNN_agent_runs()

#%%
test_trial_plot()

#%%
test_STA()

#%%
test_SR()

#%%
test_TD()



# %%
