"""
in this file, we will write a bunch of functions that are useful for identifying and analyzing slots/subspaces in our models
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys

def calc_overlap(coeffs):
    """
    take a set of decoding coeffiences and compute the overlap between subspaces.
    
    Parameters:
    ------------
    
    coeffs : torch.Tensor
        (num_slots, num_locs, num_neurons)
    """
    
    # first normalize coefficients across neurons because we want the angles not the magnitudes
    if type(coeffs) == np.ndarray:
        norm_coeffs = coeffs / np.sqrt((coeffs**2).sum(-1, keepdims = True))
    else:
        norm_coeffs = coeffs / (coeffs**2).sum(-1, keepdims = True).sqrt()
    # now compute the overlap for every pair of slots and locations
    overlap =  (coeffs[:, None, :, None, ...] * coeffs[None, :, None, ...]).sum(-1) # (num_slots, num_slots, locs, locs)
    return overlap

def calc_decoding_accs(subspace_rs, subspace_locs, coeffs, biases):
    """
    use 'coeffs' and 'biases' to predict 'subspace_locs' from 'subspace_rs' and report accuracy.
    """
    accs = []
    for isub, sub_rs in enumerate(subspace_rs): # for neural activity used to predict location in each subspace
        sub_locs = subspace_locs[isub] # true locations to predict
        pred_locs = coeffs[isub] @ sub_rs + biases[isub] # predicted distribution over locations
        accs.append((pred_locs.argmax(0) == sub_locs.argmax(0)).float().mean().item()) # did a greedy approach get it right?
    return accs

def process_subspace_trial_data(trial_data, slot_type = "relative", tmax_offset = 0):
    assert slot_type in ["planning", "relative", "absolute"]
    
    # get the data we will need
    step_nums, rs, locs, finished, num_steps = [trial_data[key] for key in ["step_nums", "rs", "locs", "finished", "num_steps"]] # (trials, steps, dim, )
    
    # look at the number of steps in each trial
    assert step_nums[..., -1].std(0).sum() == 0 # for now assume that our data array is aligned in time-within-trial
    step_nums = step_nums[0, :, 0].astype(int)
    tmin = int(step_nums[0]) # initial time
    tmax = int(step_nums[-1] + tmax_offset)
    
    # for now don't use neurons/positions at t=0 since that is at the boundary of planning and execution and might be slightly different
    if slot_type == "planning":
        # pairs of timepoints corresponding to each subspace. Here we want to decode location at time 't' from any t' in planning
        subspace_pairs = [[(t0, t1) for t0 in range(max(-2, tmin), 0)] for t1 in range(0, tmax+1)]
    elif slot_type == "relative":
        # here we want to decode location at time t+dt for any t
        subspace_pairs = [[(t0, t0+dt) for t0 in range(1, tmax+1-dt)] for dt in range(0, tmax)]
    elif slot_type == "absolute":
        # here we want to decode location at time t from any t' < t during execution
        subspace_pairs = [[(t0, t1) for t0 in range(1, t1+1)] for t1 in range(1, tmax+1)]
    else:
        raise NotImplementedError
    
    # convert from timestamps to indices within a trial based on planning time
    subspace_inds = [[(pair[0]-tmin, pair[1]-tmin) for pair in subspace]for subspace in subspace_pairs]
    num_locs = int(np.nanmax(locs)+1)
    
    # what are the neural activities used to identify each subspace?
    subspace_rs = [torch.tensor(np.concatenate([rs[:, pair[0], :] for pair in pairs], axis = 0), dtype = torch.float32).T for pairs in subspace_inds] # neural activity for all pairs of timepoints
    # what are the corresponding locations we're trying to predict?
    subspace_locs = [F.one_hot(torch.tensor(np.concatenate([locs[:, pair[1], 0] for pair in pairs], axis = 0), dtype = int), num_classes = num_locs).T for pairs in subspace_inds]
    
    return subspace_pairs, subspace_inds, subspace_rs, subspace_locs, num_locs



def find_orthogonal_subspaces(trial_data, slot_type = "relative",
                              L2_alpha = 1e-3, overlap_alpha = 2e-3, L1_alpha = 1e-4,
                              warmup = 500, max_iters = 2000, atol = 1e-3, rtol = 2e-4, lrate = 5e-3,
                              tmax_offset = 0, return_all = False):

    """
    
    Parameters
    ------------
    trial_data : dict
        dictionary with 'rs', 'step_nums', 'locs', 'finished' fields containing the data needed for subspace identification.
    slot_type : str
        how should we identify slots?
        planning: from planning data
        relative: location N actions into the future
        absolute" location at time N
    L2_alpha : float
        L2 regularization strength
    L1_alpha : float
        L1 regularization strength
    overlap_alpha : float
        strength of the overlap regularization
    warmup : int
        number of iterations to warm up for
    max_iters : int
        maximum number of iterations to run for
    tmax_offset : int
        optionally don't use data for all steps (this is useful e.g. when decoding transitions instead of states)
    """
    
    
    subspace_pairs, subspace_inds, subspace_rs, subspace_locs, num_locs = process_subspace_trial_data(trial_data, slot_type = slot_type, tmax_offset = tmax_offset)

    num_neurons = subspace_rs[0].shape[0]
    num_subspaces = len(subspace_inds) # number of subspace we are identifying
    
    print(f"identifying {num_subspaces} {slot_type} subspaces encoding {num_locs} locations from {num_neurons} neurons!")
    print(subspace_pairs)
    
    # use GPU if available since that is much faster
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    subspace_rs = [rs.to(device) for rs in subspace_rs]
    subspace_locs = [locs.to(device) for locs in subspace_locs]
    
    # we want to run logistic regression -> start from random weights and zero bias
    coeffs = nn.Parameter(torch.randn(num_subspaces, num_locs, num_neurons, device = device) / np.sqrt(num_neurons), requires_grad = True)
    biases = nn.Parameter(torch.zeros(num_subspaces, num_locs, 1, device = device), requires_grad = True)


    def calc_loss(coeffs, biases, return_all_losses = False):
        """
        loss function for the logistic regression model we train to identify slots
        """
        loss = torch.tensor(0.0, device = device)
        for ipairs, pairs in enumerate(subspace_inds): # for each pair of indices that define a subspace
            sub_rs, sub_locs = subspace_rs[ipairs], subspace_locs[ipairs]
            pred_locs = coeffs[ipairs] @ sub_rs + biases[ipairs]
            pred_locs = pred_locs - pred_locs.logsumexp(axis = 0, keepdims = True)
            loss -= (pred_locs * sub_locs).sum(0).mean() # sum across locs, mean across examples
        
        # add L2 regularization loss
        loss_reg = L2_alpha * (coeffs**2).sum()
        # add L1 reg
        loss_reg = loss_reg + L1_alpha * coeffs.abs().sum()
        
        # add orthogonality loss (normalize coeffs first to separate from L2 loss)
        overlap = calc_overlap(coeffs) # compute angles between different decoders 
        # the diagonals are just the identity, but we set it to zero anyways in case someone decides not to normalize the coefficients before regularizing
        diags = torch.diagonal(torch.diagonal(overlap, dim1=-1,dim2=-2), dim1=0, dim2=1)
        diags *= 0.0 # set diagonal to 0
        
        # we care about the absolute overlap, summed across pairs of subspaces and locations
        loss_overlap = overlap_alpha * overlap.abs().sum() 
        
        if return_all_losses:
            return loss, loss_reg, loss_overlap
        
        return loss + loss_reg + loss_overlap    
    
    # instantiate optimizer
    optim = torch.optim.Adam([coeffs, biases], lr=lrate, betas=(0.99, 0.999))
    iter_ = 0 # initialize iteration count
    old_loss, new_loss = torch.inf, torch.inf # keep track of losses
    overlap_alpha0 = overlap_alpha # final annealed overlap alpha
    losses = torch.zeros(max_iters) + torch.inf # keep track of losses
    
    # only finished if warmup is done for a bit and the loss is not changing. Also done if we exceed max_iters.
    while (iter_ < max(40, warmup*1.5)) or ((iter_ < max_iters) and (old_loss - new_loss > atol) and ((old_loss - new_loss) / loss > rtol)):
        
        warmup_scale = max(0.0, min(1.0, (iter_ - 0.2*warmup) / (0.8*warmup))) # linearly increase overlap regularization
        overlap_alpha = warmup_scale * overlap_alpha0 # linearly increase overlap_alpha from 0 to overlap_alpha0
        optim.zero_grad() # reset gradient accumulator
        loss = calc_loss(coeffs, biases) # compute loss
        loss.backward() # compute gradients
        optim.step() # update parameters
        
        # store loss
        losses[iter_] = loss.item()
        # compute running avg of losses to see if it changes
        old_loss, new_loss = losses[iter_-40:iter_-20].mean(), losses[iter_-20:iter_].mean()
        
        if iter_ % 50 == 0: # occasionally print progress
            accs = calc_decoding_accs(subspace_rs, subspace_locs, coeffs, biases)
            print(iter_, loss.item(), old_loss, new_loss, warmup_scale, accs)
            sys.stdout.flush()
        iter_ += 1 # update iteration count
    
    # print the contributions to the final loss from accuracy, L1/2 regularization, and overlap regularization
    losses = [loss.item() for loss in calc_loss(coeffs, biases, return_all_losses = True)]
    print(iter_, old_loss, new_loss, losses)
    
    # normalize the coefficients at the end since we only care about the directions in state space corresponding to the slots
    coeffs_np = coeffs.detach().cpu().numpy()
    coeffs_np_norm = coeffs_np / np.sqrt(np.sum(coeffs_np**2, -1, keepdims = True)) # normalize the weights across neurons since they should be unit vectors
    
    if return_all:
        return coeffs_np_norm, coeffs_np, biases.detach().cpu().numpy()
    else:
        return coeffs_np_norm

