
import numpy as np
import time
import pickle
import copy
import os
import sys
import torch
import pysta

def main_train(kwargs):
    
    # print arguments
    print("\n\nSetting up model for training:")
    print(kwargs)

    # set some parameters
    np.random.seed(kwargs["seed"])
    torch.manual_seed(kwargs["seed"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # instantiate model
    env = pysta.envs.MazeEnv(**kwargs)
    rnn = pysta.agents.VanillaRNN(env, **kwargs).to(device)

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
    all_losses, all_accs = [], []
    best_loss = np.inf

    # print training message
    time.sleep(5e-2)
    print(f"Training {kwargs['num_epochs']} batches of size {rnn.env.batch} on {device}")

    # now run actual training loop
    t0 = time.time()
    for epoch in range(kwargs["num_epochs"]):
        
        if epoch % kwargs["eval_freq"] == 0:
            with torch.no_grad():
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
                    pickle.dump({"epoch": epoch, "loss": all_losses, "accs": all_accs, "rnn": rnn, "best_loss": best_loss, "kwargs": kwargs, "optim": optim}, open(f"{savename}.p", "wb"))
                
        optim.zero_grad() # reset gradient accumulator
        loss = rnn.forward() # compute loss
        loss.backward() # compute gradients
        optim.step() # update parameters
        
    if kwargs["save_results"]:
        pickle.dump({"epoch": epoch, "loss": all_losses, "accs": all_accs, "rnn": rnn, "best_loss": best_loss, "kwargs": kwargs, "optim": optim}, open(f"{savename}.p", "wb"))
        torch.save(rnn, f"{savename}_final.pt")

    return rnn

if __name__ == "__main__":
    kwargs = pysta.argparser.parse_args()
    main_train(kwargs)
