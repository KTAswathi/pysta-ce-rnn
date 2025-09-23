"""Code for testing that the RNN initialisation and training runs"""

#%%
import pysta
import torch
import numpy as np

def test_training_runs():
    print("\nTest that training runs")
    pysta.reload()
    kwargs = pysta.argparser.parse_args()
    kwargs["save_results"] = False
    kwargs["batch_size"] = 10
    kwargs["num_epochs"] = 20
    kwargs["num_eval"] = 1
    kwargs["eval_freq"] = 5
    kwargs["overwrite"] = True
    kwargs["prefix"] = "test_"
        
    pysta.train_rnn.main_train(kwargs)
    
    print("Done!")
    return

#%%
print("\n\nTesting training script!")

#%%
test_training_runs()



# %%
