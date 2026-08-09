
import numpy as np
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections.abc import Mapping
from pathlib import Path
from pysta.embedding.mpfc_embedding import load_embedding

#%% base agent
class BaseAgent(nn.Module):
    classname = "BaseAgent"
    label = "base"

    def __init__(self, env, rec_noise = 1e-3, ent_reg = 1e-5, greedy = False, force_optimal = False, iters_per_action = 1, tau = 1, **kwargs):
        """
        RNN agent that learns to navigate in a maze

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        rec_noise : float
            Noise added to the recurrent dynamics
        ent_reg: float
            Amount of entropy regularization to add
        greedy : bool
            If True, take greedy actions, otherwise sample from the policy
        force_optimal : bool
            Only allow optimal actions. If there are multiple actions, renormalize the policy over these.
        iters_per_action : int or list of ints
            Number of RNN iterations to run for each environment iteration.
        tau : float
            timescale of the dynamics. r_{t+1} = (1-1/tau) * r_t + (1/tau) * f(r_t, x_{t+1})
        """
        
        # pytorch boilerplate
        super(BaseAgent, self).__init__()

        # store some hyperparameters
        self.env = env
        self.greedy = greedy
        self.force_optimal = force_optimal
        self.iters_per_action = iters_per_action
        self.tau = tau
        self.rec_noise = rec_noise
        self.store_all_activity = False
        self.ent_reg = ent_reg
    
        # and get some from the environment
        self.Nout = self.env.output_dim # dimensionality of the policy we're learning
        self.Nin = self.env.obs_dim # dimensionality of the observations

        # instantiate parameters
        self.initialise_weights()
        
        # reset the model to its initial conditions
        self.reset()
        
        return
        
    def initialise_weights(self):
        """
        Instantiate the model parameters
        """
        raise NotImplementedError
    
    def allo_to_ego_pi(self, allo_pi):
        """
        Convert an allocentric policy to an egocentric policy by renormalizing over neighboring states
        In this version, each action is given the non-normalized probability associated with the resulting state (instead of distributing the probability mass)
        This is done to ensure that the greedy actions agree
        """
        
        ego_pi = torch.zeros(allo_pi.shape[0], self.env.num_actions) # new egocentric policy
        new_locs = self.env.neighbors[self.env.batch_inds, self.env.loc, :] # for each action, where do I end up
        for a in range(self.env.num_actions): # for each action
            new_locs_a = new_locs[:, a] # where would I end up
            ego_pi[:, a] = allo_pi[self.env.batch_inds, new_locs_a] # what is the probability of going here
        return ego_pi / ego_pi.sum(-1, keepdims = True)
        
    @property
    def name(self):
        """Generates a string representation of the agent"""
        if type(self.iters_per_action) in [int, np.int32, np.int64]:
            iter_str = f"iter{self.iters_per_action}"
        else:
            iter_str = f"iter{'-'.join([str(val) for val in self.iters_per_action])}"

        tau_str = f"tau{self.tau}"
        force_str = "opt" if self.force_optimal else "agent"
        return f"{self.classname}/{iter_str}_{tau_str}_{force_str}"

    def reset(self):
        """
        Reset the agent.
        This function sets the initial condition and empties caches storing data.
        """
        
        # no gradients for resetting environment
        with torch.no_grad():
            self.env.reset() # reset environment state

            # also initialise some lists to store data along the way
            self.store = [] # store many things at the time of each action
            self.all_acts = [[], [], []] # also store some information for dynamics in between actions
            
        # need gradients for setting initial z
        self.z = torch.zeros(torch.Size([self.env.batch])+self.z0.shape, device = self.z0.device) + self.z0[None, ...]
        self.r = self.phi(self.z)
        
        # instantiate loss functions
        self.acc_loss = torch.tensor(0.0, device=self.z0.device) # accuracy
        self.weight_loss = self.env.batch * self.calc_parameter_reg() # parameter regularization
        self.rate_loss = self.calc_activity_reg() # rate regularization
        self.ent_loss = torch.tensor(0.0, device=self.z0.device) # entropy regularization
        self.update_optimal_actions() # cache optimal actions
        
        return

    def _as_batch_mask(self, mask, name, device):
        """Convert an environment mask to a boolean tensor of shape ``(batch,)``."""
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if mask.ndim == 0:
            mask = mask.expand(self.env.batch)
        if tuple(mask.shape) != (self.env.batch,):
            raise ValueError(
                f"{name} must have shape ({self.env.batch},), got {tuple(mask.shape)}."
            )
        return mask

    def _policy_loss_mask(self, device=None):
        """Return the trials whose current action contributes to policy metrics/loss."""
        device = self.z0.device if device is None else device
        policy_loss_mask = getattr(self.env, "policy_loss_mask", None)
        if callable(policy_loss_mask):
            return self._as_batch_mask(
                policy_loss_mask(), "env.policy_loss_mask()", device
            )

        # Jensen MazeEnv fallback: execution phase and unfinished trials only.
        not_finished = self._as_batch_mask(
            ~torch.as_tensor(self.env.finished, dtype=torch.bool),
            "~env.finished",
            device,
        )
        step_num = torch.as_tensor(self.env.step_num, device=device)
        execution_phase = self._as_batch_mask(
            step_num >= 0, "env.step_num >= 0", device
        )
        return execution_phase & not_finished

    def _action_sampling_mask(self, device, dtype):
        """Return an optional per-trial mask over actions available for sampling."""
        action_sampling_mask = getattr(self.env, "action_sampling_mask", None)
        if callable(action_sampling_mask):
            mask = action_sampling_mask()
            if mask is None:
                return None
        elif getattr(self.env, "output_format", None) == "allocentric":
            # Jensen MazeEnv fallback: allocentric outputs denote destination states,
            # so only adjacent states may be sampled.
            mask = self.env.adjacency[
                self.env.batch_inds, self.env.loc, :
            ]
            assert mask.sum(-1).min() >= 2
        else:
            # Jensen egocentric policies already enumerate valid actions.
            return None

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        if mask.ndim == 1 and tuple(mask.shape) == (self.Nout,):
            mask = mask[None, :].expand(self.env.batch, -1)
        if tuple(mask.shape) != (self.env.batch, self.Nout):
            raise ValueError(
                "env.action_sampling_mask() must have shape "
                f"({self.env.batch}, {self.Nout}), got {tuple(mask.shape)}."
            )
        if not torch.isfinite(mask).all() or torch.any(mask < 0):
            raise ValueError("Action-sampling masks must be finite and non-negative.")
        return mask

    @staticmethod
    def _snapshot_value(value):
        """Copy mutable environment metadata before the environment advances again."""
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, Mapping):
            return {
                key: BaseAgent._snapshot_value(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [BaseAgent._snapshot_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(BaseAgent._snapshot_value(item) for item in value)
        return value

    def _environment_metadata(self, hook_name):
        """Read and snapshot optional environment-specific trajectory metadata."""
        hook = getattr(self.env, hook_name, None)
        if not callable(hook):
            return {}
        metadata = hook()
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise TypeError(f"env.{hook_name}() must return a mapping or None.")
        return {
            key: self._snapshot_value(value) for key, value in metadata.items()
        }
    
    def sample_action(self):
        """
        This function samples an action for each trial in a batch from the current policy.
        Note that 'self.action' is the action chosen by an unconstrained agent.
        'self.env_action' is the action passed to the environment and can be different if we enforce optimality.
        We distinguish between the two because e.g. accuracies are still computed from the sampled action.

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch

        """
        
        with torch.no_grad():
            pi = self.pi
            sampling_mask = self._action_sampling_mask(pi.device, pi.dtype)
            active = self._policy_loss_mask(pi.device)

            if sampling_mask is not None:
                # Add jitter only to permitted actions. Rows outside the policy-loss
                # phase may legitimately expose an empty mask because their action is
                # ignored; retain the unconstrained policy for those rows so sampling
                # remains numerically defined.
                masked_pi = (pi + 1e-20) * sampling_mask
                normalizer = masked_pi.sum(-1, keepdim=True)
                valid_rows = normalizer[..., 0] > 0
                if torch.any(active & ~valid_rows):
                    raise ValueError(
                        "env.action_sampling_mask() permits no actions for an active trial."
                    )
                normalized_masked_pi = masked_pi / normalizer.clamp_min(
                    torch.finfo(pi.dtype).tiny
                )
                pi = torch.where(valid_rows[:, None], normalized_masked_pi, pi)
                
            if self.greedy: # pick the most likely action
                self.action = torch.argmax(pi, -1)
            else: # sample an action
                try:
                    self.action = torch.multinomial(pi, 1)[..., 0]
                except RuntimeError:
                    # something went wrong, so we save the current state of the agent for debugging.
                    # This may happen if the policy becomes near-deterministic, and it can often be resolved by increasing the entropy regularisation.
                    print(pi.min(), pi.max(), self.pi.min(), self.pi.max(), self.logpi.min(), self.logpi.max())
                    pickle.dump(self, open("./temp.p", "wb"))
                    raise
                
            # Outside the policy-loss phase the environment ignores the action. Start
            # from the unconstrained sample, then enforce optimality only for active rows.
            self.env_action = self.action.clone()
            teacher_rows = active
            if not callable(getattr(self.env, "policy_loss_mask", None)):
                # Exact Jensen fallback: the old agent teacher-forced every row,
                # including planning/finished rows whose actions were ignored.
                teacher_rows = torch.ones_like(active)
            if self.force_optimal and torch.any(teacher_rows):
                optimal_actions = torch.as_tensor(
                    self.optimal_actions, dtype=pi.dtype, device=pi.device
                )
                if tuple(optimal_actions.shape) != tuple(pi.shape):
                    raise ValueError(
                        "env.optimal_actions() must have shape "
                        f"{tuple(pi.shape)}, got {tuple(optimal_actions.shape)}."
                    )

                opt_pis = pi[teacher_rows] * optimal_actions[teacher_rows]
                jitter = torch.rand_like(opt_pis) * 1e-20
                if sampling_mask is not None:
                    jitter = jitter * (sampling_mask[teacher_rows] > 0).to(pi.dtype)
                opt_pis = opt_pis + jitter

                if torch.any(opt_pis.sum(-1) <= 0):
                    raise ValueError("No optimal action is available for an active trial.")
                if self.greedy: # pick most likely action
                    optimal_samples = torch.argmax(opt_pis, -1)
                else: # sample an action
                    optimal_samples = torch.multinomial(opt_pis, 1)[..., 0]
                self.env_action[teacher_rows] = optimal_samples
            
        return self.action
    
    def step(self, observation):
        """
        Perform all the computations happening before the next environment update step
        This can include multiple iterations of recurrent network dynamics.

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """
        
        raise NotImplementedError
    
    def calc_activity_reg(self, not_finished = None):
        """
        Compute firing rate regularization loss. 0 by default.
        """
        return 0.0
    
    def calc_parameter_reg(self):
        """
        Compute weight regularization loss. 0 by default.
        """
        return 0.0

    def update_loss(self, loss_mask=None):
        """
        Accumulate losses.
        Treat accuracy loss, rate loss, and parameter loss separately.
        Losses are only added for trials that have not finished.
        """
        
        loss_mask = (
            self._policy_loss_mask(self.pi.device)
            if loss_mask is None
            else self._as_batch_mask(loss_mask, "loss_mask", self.pi.device)
        )

        if torch.any(loss_mask):
            # policy loss (sum_{a \in opt_as} pi(a))
            optimal_actions = torch.as_tensor(
                self.optimal_actions, dtype=self.pi.dtype, device=self.pi.device
            )
            opt_probs = (self.pi * optimal_actions)[loss_mask, :].sum(-1)
            assert opt_probs.max() < 1.0 + 1e-5 # check that things are not too crazy
            self.acc_loss = self.acc_loss + (1.0 - opt_probs).sum() # turn our objective into a loss and sum across batches
            
            # entropy loss
            jitter = 1e-5 # add a little bit of jitter to avoid nans
            pi_ent = (self.pi + jitter) / (1+self.pi.shape[-1] * jitter) # make sure things are normalised
            self.ent_loss = self.ent_loss + self.ent_reg * (pi_ent*pi_ent.log())[loss_mask, :].sum() # want to maximize entropy; minimize -H = E[pi logpi]
    
        return
    
    def update_store(self, loss_mask=None, observation=None):
        """
        Update list of environment/agent states to include the current state.
        """
        
        loss_mask = (
            self._policy_loss_mask(self.action.device)
            if loss_mask is None
            else self._as_batch_mask(loss_mask, "loss_mask", self.action.device)
        )
        optimal_actions = torch.as_tensor(
            self.optimal_actions, dtype=self.pi.dtype, device=self.action.device
        )
        batch_inds = torch.arange(self.env.batch, device=self.action.device)
        corrects = optimal_actions[batch_inds, self.action].clone()
        if callable(getattr(self.env, "policy_loss_mask", None)):
            corrects[~loss_mask] = torch.nan
        else:
            # Preserve Jensen's stored planning correctness values; only
            # already-finished trials were NaN in the original store format.
            finished = self._as_batch_mask(
                self.env.finished, "env.finished", self.action.device
            )
            corrects[finished] = torch.nan
        if observation is None:
            observation = self.env.observation().to(self.z0.device)

        record = {
            "rs": self.r.detach(), # firing rate
            "zs": self.z.detach(), # neural potential
            "action": self.action, # action generated by the agent
            "env_action": self.env_action, # action passed to the environment
            "optimal_actions": self.optimal_actions, # optimal action
            "finished": self._snapshot_value(self.env.finished), # which trials have finished
            "loss_mask": loss_mask.detach().clone(), # which actions contribute to policy loss/accuracy
            "pi": self.pi.detach(), # policy
            "xs": observation.detach(), # exact inputs used for this RNN/environment step
            "corrects": corrects, # whether actions are correct
        }

        # Keep Jensen's established fields when the environment exposes them,
        # without requiring them from generic environments.
        if hasattr(self.env, "loc"):
            record["loc"] = self._snapshot_value(self.env.loc)
        if hasattr(self.env, "step_num"):
            record["step_num"] = self._snapshot_value(self.env.step_num)

        record.update(self._environment_metadata("trajectory_metadata"))
        self.store.append(record)

    def update_post_step_store(self):
        """Attach optional post-transition metadata to the most recent record."""
        if len(self.store) == 0:
            return
        self.store[-1].update(self._environment_metadata("post_step_metadata"))
        
    def update_optimal_actions(self):
        """
        Cache optimal actions
        """
        with torch.no_grad():
            self.optimal_actions = torch.as_tensor(
                self.env.optimal_actions(), dtype=self.z0.dtype, device=self.z0.device
            ) # tensor (batch, output_dim)

    def forward(self, store = False):
        """
        Run a single batch of trials

        Parameters
        ----------
        store : bool
            if true, store environment and agent states after every action
            
        Returns
        ----------
        avg_loss : tensor
            average loss across trials in a batch
        """
        
        self.reset() # reset agent
        
        while not torch.all(self.env.finished): # as long as there are some trials left to act in
            self.update_optimal_actions() # cache the optimal actions at the current location
            
            x = self.env.observation().to(self.z0.device) # observation at this point in time
            self.step(x) # update RNN state, compute policy, and sample an action
            loss_mask = self._policy_loss_mask(self.pi.device)
            self.update_loss(loss_mask) # update performance and entropy loss
            
            # now update environment and optionally store env+agent state (don't propagate gradients through this)
            with torch.no_grad():
                if store:
                    self.update_store(loss_mask, observation=x)
                self.env.step(self.env_action) # action passed to the environment (optionally restricted to be optimal)
                if store:
                    self.update_post_step_store()
        
        # loss is combined accuracy and regularization losses, normalized by the batch size
        return (self.acc_loss + self.weight_loss + self.rate_loss + self.ent_loss) / self.env.batch

    def eval(self, num_eval = 5):
        """
        Compute average loss and accuracy across a number of trials

        Parameters
        ----------
        num_eval : int
            number of batches to average over
            
        Returns
        ----------
        avg_loss : float
            average loss across trials in all batches
        avg_acc : float
            average accuracy across trials in all batches
        """
        losses, accs = [], [] # lists to concatenate across batches
        for _ in range(num_eval): # for each batch
            # simulate the trials
            loss = self.forward(store = True).detach().cpu().numpy()
            losses.append(loss) # append loss
            corrects = torch.stack([s["corrects"] for s in self.store])
            loss_masks = torch.stack([s["loss_mask"] for s in self.store]).to(
                device=corrects.device, dtype=torch.bool
            )
            corrects = corrects.masked_fill(~loss_masks, torch.nan)
            per_trial_acc = torch.nanmean(corrects, dim=0)
            accs.append(torch.nanmean(per_trial_acc).detach().cpu().item())
            
        return np.mean(losses), np.mean(accs)
            
    def plot_trial(self, filename = None, run_trial = True, trial_num = 0, values = True, vmap = None, cmap = "coolwarm"):
        """
        Plot a summary plot for a trial

        Parameters
        ----------
        filename : str
            filename to save to
        run_trial : bool
            whether to run a new trial (True) or use cached information (False)
        trial_num : int
            which trial number within the batch to plot
        values : bool
            for the reward landscape task, whether to plot a value map (True) or reward map (False)
        vmap : tensor
            optional data to plot as a heatmap for each time point
        cmap : str
            colormap to use
        """
        
        if run_trial: # optionally simulate a new trial
            self.forward(store = True)
        
        # find out which time points have not finished
        fs = 1 - np.array([store["finished"][trial_num].numpy() for store in self.store])
        T = min(np.sum(fs)+1, len(self.store)) # number of time points to plot
        fig, axs = plt.subplots(1, T, figsize = (2*T, 2))
        for t, ax in enumerate(axs): # for each time point
            data = self.store[t] # get data for this time point
            # plot a panel
            vmap_t = None if vmap is None else vmap[t] # optionally plot some useful data
            self.env.plot(loc = data["loc"][trial_num], step_num = data["step_num"], ax = ax, trial_num = trial_num, values = values, cmap = cmap, vmap = vmap_t)
        plt.tight_layout = True
        if filename is not None:
            plt.savefig(filename, bbox_inches = "tight")
        plt.close()
        
        return


#%% Vanilla RNN

class VanillaRNN(BaseAgent):
    classname = "VanillaRNN"
    label = "rnn"
    
    def __init__(self, env, Nrec = 800, W_reg = 1e-3, r_reg = 1e-3, nonlin_output = False, **kwargs):
        """
        RNN agent that learns to navigate in a maze

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        Nrec : int
            Number of relu units in the RNN's hidden layer
        W_reg : float
            Regularization strength for the parameters
        r_reg : float
            Regularization strength for the hidden layer activity
        nonlin_output : bool
            If True, have an additional hidden layer in the readout. Otherwise use a linear readout from the hidden state.
        """
    
        # store some model-specific parameters
        self.Nrec = Nrec
        self.nonlin_output = nonlin_output
        self.W_reg, self.r_reg = W_reg, r_reg 
        self.phi = F.relu # use a ReLU nonlinearity
        
        # initialise BaseAgent
        super(VanillaRNN, self).__init__(env, **kwargs)
        
    @property
    def name(self):
        """generates a string representation of the agent"""
        nonlin_str = "nonlinout" if self.nonlin_output else "linout"
        basename = super(VanillaRNN, self).name
        return f"{basename}/N{self.Nrec}_{nonlin_str}"
    
    def initialise_weights(self):
        """
        Instantiate the learnable model parameters.
        We just initialise the parameters as iid Gaussian variables.
        """

        # RNN initial condition
        self.z0 = nn.Parameter(torch.randn(self.Nrec, 1), requires_grad=True)
        # recurrent weight matrix
        self.Wrec = nn.Parameter(torch.randn(self.Nrec, self.Nrec) / np.sqrt(self.Nrec), requires_grad=True)
        # input weight matrix
        self.Win = nn.Parameter(torch.randn(self.Nrec, self.Nin) / np.sqrt(self.Nin), requires_grad=True)
        # hidden state bias
        self.brec = nn.Parameter(torch.zeros(self.Nrec, 1), requires_grad=True)
        
        # now initialise the output function, which can be either nonlinear or linear
        if self.nonlin_output:
            # create one hidden layer between the RNN and policy

            # weights and bias to hidden output layer
            self.Wout1 = nn.Parameter(torch.randn(int(np.round(self.Nrec/2)), self.Nrec) / np.sqrt(self.Nrec), requires_grad=True)
            self.bout1 = nn.Parameter(torch.zeros(self.Wout1.shape[0], 1), requires_grad=True)

            # weights and bias to policy
            self.Wout = nn.Parameter(torch.randn(self.Nout, self.Wout1.shape[0]) / np.sqrt(self.Wout1.shape[0]), requires_grad=True)
            self.bout = nn.Parameter(torch.zeros(self.Nout, 1), requires_grad=True)

        else:
            # just a single weight and bias for a linear readout to out policy
            self.Wout = nn.Parameter(torch.randn(self.Nout, self.Nrec) / np.sqrt(self.Nrec), requires_grad=True)
            self.bout = nn.Parameter(torch.zeros(self.Nout, 1), requires_grad=True)
        
        return

    def step(self, observation):
        """
        Perform all the computations happening before the next environment update step
        This can include multiple iterations of recurrent network dynamics.

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """
        
        batch = observation.shape[0] # batch size
        
        # decide how many network iterations to run for this environment iteration
        network_iters = self.iters_per_action if type(self.iters_per_action) in [int, np.int32, np.int64] else np.random.choice(self.iters_per_action)
        for _ in range(network_iters): # optionally several network iterations
            
            # recurrent noise
            rec_noise = torch.randn(batch, self.Nrec, 1, device = self.z0.device) * self.rec_noise
            
            # feedforward input
            ff_inp = self.Win @ observation[..., None]
            
            # recurrent input
            rec_inp =  self.Wrec @ self.r

            # update neuron potentials
            self.z = (1 - 1/self.tau) * self.z + (1/self.tau)*(rec_inp + ff_inp + self.brec + rec_noise)
            
            # compute firing rates
            self.r = self.phi(self.z) # firing rate
            
            # update rate loss for trials that have not finished
            self.rate_loss = self.rate_loss + self.calc_activity_reg(torch.where(~self.env.finished)[0])
            
            if self.store_all_activity: # optinoally store activity at every RNN iteration
                self.all_acts[0].append(self.r.detach().numpy())
                self.all_acts[1].append( self.env.loc.detach().numpy())
                self.all_acts[2].append(self.env.step_num)
        
        # compute output
        if self.nonlin_output:
            rout = self.phi(self.Wout1 @ self.r + self.bout1) # pass through hidden layer
        else:
            rout = self.r # just linear readout
        self.logpi = (self.Wout @ rout + self.bout)[..., 0] # flatten for each batch (batch x actions x 1) -> (batch, actions)

        # normalize log policy
        self.logpi = self.logpi - self.logpi.logsumexp(-1, keepdims = True)
        self.pi = self.logpi.exp() # compute policy (batch, actions)
        
        # sample an action from the policy
        self.action = self.sample_action()
        
        return self.action
    
    def calc_activity_reg(self, not_finished = None):
        """
        Compute firing rate regularization loss

        Parameters
        ----------
        not_finished : bool
            trials within the batch which have not yet finished.
            if None: assume no trials are finished.

        Returns
        ----------
        reg_loss : tensor
            firing rate regularization loss. Summed across trials and neurons
        """
        
        if not_finished is None:
            reg_loss = self.r_reg * (self.r**2).sum() # magnitude of hidden state vector
        else:
            reg_loss = self.r_reg * (self.r[not_finished]**2).sum() # magnitude of hidden state vector
        return reg_loss
    
    def calc_parameter_reg(self):
        """
        Compute weight regularization loss
            
        Returns
        ----------
        reg_loss : tensor
            weight regularization loss. sum across all parameters.
        """
        
        reg_loss = self.W_reg * torch.stack([torch.square(p).sum() for p in self.parameters()]).sum() # magnitude of total weight vecto

        return reg_loss
    
# line-embedded version of VanillaRNN with optional local input/output anatomy
class LineEmbeddedRNN(VanillaRNN):
    classname = "LineEmbeddedRNN"
    label = "line_rnn"

    def __init__(
        self,
        env,
        Nrec=800,
        W_reg=1e-3,
        r_reg=1e-3,
        nonlin_output=False,
        dist_reg=1e-7,
        line_decay=0.12,
        line_init_scale=1.5,
        use_local_init=True,
        localize_loc_input=False,
        localize_rew_input=False,
        localize_wall_input=False,
        local_fraction=1.0 / 6.0,
        readout_mode="global",
        **kwargs,
    ):
        # geometry-related hyperparameters
        self.dist_reg = dist_reg
        self.line_decay = line_decay
        self.line_init_scale = line_init_scale
        self.use_local_init = use_local_init

        # local input/ output options
        self.localize_loc_input = localize_loc_input
        self.localize_rew_input = localize_rew_input
        self.localize_wall_input = localize_wall_input
        self.local_fraction = local_fraction
        self.readout_mode = readout_mode

        if self.readout_mode not in ["global", "same_end", "opposite_end"]:
            raise ValueError(
                f"readout_mode must be one of ['global', 'same_end', 'opposite_end'], got {self.readout_mode}"
            )

        super(LineEmbeddedRNN, self).__init__(
            env,
            Nrec=Nrec,
            W_reg=W_reg,
            r_reg=r_reg,
            nonlin_output=nonlin_output,
            **kwargs,
        )

    @property
    def name(self):
        # append line-embedding + local I/O details
        nonlin_str = "nonlinout" if self.nonlin_output else "linout"
        basename = super(VanillaRNN, self).name
        return (
            f"{basename}/N{self.Nrec}_{nonlin_str}"
            f"_line_ld{self.line_decay}_dr{self.dist_reg}"
            f"_lfrac{self.local_fraction}"
            f"_loc{int(self.localize_loc_input)}"
            f"_rew{int(self.localize_rew_input)}"
            f"_wall{int(self.localize_wall_input)}"
            f"_ro{self.readout_mode}"
        )

    # helper: how many units belong to the local band
    def _local_band_size(self):
        return max(1, int(np.round(self.local_fraction * self.Nrec)))

    # helper: first k units = same_end, last k units = opposite_end
    def _make_unit_band_mask(self, mode):
        mask = torch.zeros(self.Nrec, dtype=torch.float32)
        k = self._local_band_size()

        if mode == "global":
            mask[:] = 1.0
        elif mode == "same_end":
            mask[:k] = 1.0
        elif mode == "opposite_end":
            mask[-k:] = 1.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return mask

    # helper: build input routing masks from env.obs_inds()
    def _build_input_masks(self):
        inds = self.env.obs_inds()
        if not isinstance(inds, Mapping):
            raise TypeError("env.obs_inds() must return a mapping of groups to indices.")

        routing_hook = getattr(self.env, "input_routing", None)
        routing_hook_name = "input_routing"
        if not callable(routing_hook):
            routing_hook = getattr(self.env, "cortical_input_routing", None)
            routing_hook_name = "cortical_input_routing"

        if callable(routing_hook):
            routing = routing_hook()
            if not isinstance(routing, Mapping):
                raise TypeError(
                    f"env.{routing_hook_name}() must return a mapping of groups to modes."
                )
            buffer_names = {group: f"mask_{group}" for group in inds}
        else:
            # Exact Jensen MazeEnv fallback, retaining its original flags and
            # public buffer names for existing models and analyses.
            expected_groups = {"loc", "goal", "walls"}
            if set(inds) != expected_groups:
                raise ValueError(
                    "An environment without input_routing() or "
                    "cortical_input_routing() must expose Jensen MazeEnv groups "
                    f"{sorted(expected_groups)}, got {sorted(inds)}."
                )
            routing = {
                "loc": "local" if self.localize_loc_input else "global",
                "goal": "local" if self.localize_rew_input else "global",
                "walls": "local" if self.localize_wall_input else "global",
            }
            buffer_names = {
                "loc": "mask_loc",
                "goal": "mask_rew",
                "walls": "mask_wall",
            }

        if set(routing) != set(inds):
            missing = sorted(set(inds) - set(routing))
            extra = sorted(set(routing) - set(inds))
            raise ValueError(
                "Input-routing groups must exactly match env.obs_inds(); "
                f"missing={missing}, extra={extra}."
            )

        combined_mask = torch.zeros(self.Nrec, self.Nin, dtype=torch.float32)
        covered_inputs = torch.zeros(self.Nin, dtype=torch.bool)
        input_mask_buffers = {}
        input_routing_modes = {}

        mode_aliases = {
            "local": "same_end",
            "same_end": "same_end",
            "global": "global",
            "opposite": "opposite_end",
            "opposite_end": "opposite_end",
        }
        unit_masks = {
            "same_end": self.same_end_unit_mask,
            "global": torch.ones(self.Nrec, dtype=torch.float32),
            "opposite_end": self.opposite_end_unit_mask,
        }

        for group, group_inds in inds.items():
            if not isinstance(group, str):
                raise TypeError("env.obs_inds() group names must be strings.")

            group_inds = torch.as_tensor(group_inds)
            if group_inds.ndim != 1:
                raise ValueError(
                    f"Observation indices for group '{group}' must be one-dimensional."
                )
            if group_inds.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise TypeError(
                    f"Observation indices for group '{group}' must be integers."
                )
            group_inds = group_inds.to(dtype=torch.long)
            if group_inds.numel() != torch.unique(group_inds).numel():
                raise ValueError(
                    f"Observation group '{group}' contains duplicate indices."
                )
            if group_inds.numel() > 0 and (
                torch.any(group_inds < 0) or torch.any(group_inds >= self.Nin)
            ):
                raise ValueError(
                    f"Observation group '{group}' contains indices outside 0..{self.Nin - 1}."
                )
            if torch.any(covered_inputs[group_inds]):
                raise ValueError(
                    f"Observation group '{group}' overlaps another input group."
                )

            mode = routing[group]
            if mode not in mode_aliases:
                raise ValueError(
                    f"Unknown routing mode '{mode}' for group '{group}'; expected "
                    "'local'/'same_end', 'global', or 'opposite'/'opposite_end'."
                )
            normalized_mode = mode_aliases[mode]
            group_mask = torch.zeros(self.Nrec, self.Nin, dtype=torch.float32)
            group_mask[:, group_inds] = unit_masks[normalized_mode][:, None]

            buffer_name = buffer_names[group]
            if not buffer_name.isidentifier() or buffer_name == "mask_input":
                raise ValueError(
                    f"Observation group '{group}' cannot be registered as buffer '{buffer_name}'."
                )
            self.register_buffer(buffer_name, group_mask)
            input_mask_buffers[group] = buffer_name
            input_routing_modes[group] = normalized_mode
            combined_mask = combined_mask + group_mask
            covered_inputs[group_inds] = True

        if not torch.all(covered_inputs):
            missing_inputs = torch.where(~covered_inputs)[0].tolist()
            raise ValueError(
                "env.obs_inds() must cover every observation channel exactly once; "
                f"missing indices={missing_inputs}."
            )

        self.input_mask_buffers = input_mask_buffers
        self.input_routing_modes = input_routing_modes
        self.register_buffer("mask_input", combined_mask)

    def initialise_weights(self):
        """
        Same overall architecture as VanillaRNN, but recurrent weights are
        initialised with a distance-biased random pattern on a 1D line.
        Inputs and readout can optionally be localized.
        """

        # RNN initial condition
        self.z0 = nn.Parameter(torch.randn(self.Nrec, 1), requires_grad=True)

        # 1D line embedding: unit positions and distance matrix
        positions = torch.linspace(0.0, 1.0, steps=self.Nrec)
        D = torch.abs(positions[:, None] - positions[None, :])

        # rescale distances so the regulariser is numerically well-behaved
        D = D / D.mean().clamp(min=1e-8)

        self.register_buffer("unit_positions", positions)
        self.register_buffer("distance_matrix", D)

        # define same-end and opposite-end bands once, for reuse in input/output routing
        self.register_buffer("same_end_unit_mask", self._make_unit_band_mask("same_end"))
        self.register_buffer("opposite_end_unit_mask", self._make_unit_band_mask("opposite_end"))

        if self.readout_mode == "global":
            readout_mask = self._make_unit_band_mask("global")
        elif self.readout_mode == "same_end":
            readout_mask = self._make_unit_band_mask("same_end")
        elif self.readout_mode == "opposite_end":
            readout_mask = self._make_unit_band_mask("opposite_end")
        else:
            raise ValueError(f"Unknown readout_mode: {self.readout_mode}")

        self.register_buffer("readout_unit_mask", readout_mask)

        # nearby units connect more strongly at init
        locality = torch.exp(-self.distance_matrix / self.line_decay)

        # signed random recurrent weights, modulated by locality
        Wrec = (
            self.line_init_scale
            * torch.randn(self.Nrec, self.Nrec)
            / np.sqrt(self.Nrec)
        )
        if self.use_local_init:
            Wrec = Wrec * locality

        self.Wrec = nn.Parameter(Wrec, requires_grad=True)

        # input matrix is still learnable, but routing will be controlled by masks
        self.Win = nn.Parameter(
            torch.randn(self.Nrec, self.Nin) / np.sqrt(self.Nin),
            requires_grad=True,
        )

        # hidden state bias
        self.brec = nn.Parameter(torch.zeros(self.Nrec, 1), requires_grad=True)

        # build input routing masks after Win/Nin/Nrec are known
        self._build_input_masks()

        # output layer stays exactly like VanillaRNN
        if self.nonlin_output:
            self.Wout1 = nn.Parameter(
                torch.randn(int(np.round(self.Nrec / 2)), self.Nrec)
                / np.sqrt(self.Nrec),
                requires_grad=True,
            )
            self.bout1 = nn.Parameter(
                torch.zeros(self.Wout1.shape[0], 1),
                requires_grad=True,
            )

            self.Wout = nn.Parameter(
                torch.randn(self.Nout, self.Wout1.shape[0])
                / np.sqrt(self.Wout1.shape[0]),
                requires_grad=True,
            )
            self.bout = nn.Parameter(torch.zeros(self.Nout, 1), requires_grad=True)

        else:
            self.Wout = nn.Parameter(
                torch.randn(self.Nout, self.Nrec) / np.sqrt(self.Nrec),
                requires_grad=True,
            )
            self.bout = nn.Parameter(torch.zeros(self.Nout, 1), requires_grad=True)

        return

    def step(self, observation):
        """
        Perform all computations before the next environment update step.
        Same as VanillaRNN, but with optional local input routing and optional local/global readout.
        """

        batch = observation.shape[0]

        network_iters = (
            self.iters_per_action
            if type(self.iters_per_action) in [int, np.int32, np.int64]
            else np.random.choice(self.iters_per_action)
        )

        for _ in range(network_iters):
            rec_noise = torch.randn(batch, self.Nrec, 1, device=self.z0.device) * self.rec_noise

            # Apply the validated combined routing mask. Whole-object Jensen
            # checkpoints created before generic routing have the three legacy
            # masks but no ``mask_input`` buffer, so retain a read-only fallback.
            obs_col = observation[..., None]  # (batch, Nin, 1)
            if hasattr(self, "mask_input"):
                input_mask = self.mask_input
            else:
                input_mask = self.mask_loc + self.mask_rew + self.mask_wall
            ff_inp = (self.Win * input_mask) @ obs_col

            rec_inp = self.Wrec @ self.r

            self.z = (1 - 1 / self.tau) * self.z + (1 / self.tau) * (
                rec_inp + ff_inp + self.brec + rec_noise
            )

            self.r = self.phi(self.z)

            self.rate_loss = self.rate_loss + self.calc_activity_reg(
                torch.where(~self.env.finished)[0]
            )

            if self.store_all_activity:
                self.all_acts[0].append(self.r.detach().numpy())
                self.all_acts[1].append(self.env.loc.detach().numpy())
                self.all_acts[2].append(self.env.step_num)

        # restrict which units contribute to the readout
        masked_r = self.r * self.readout_unit_mask[None, :, None]

        if self.nonlin_output:
            rout = self.phi(self.Wout1 @ masked_r + self.bout1)
        else:
            rout = masked_r

        self.logpi = (self.Wout @ rout + self.bout)[..., 0]

        self.logpi = self.logpi - self.logpi.logsumexp(-1, keepdims=True)
        self.pi = self.logpi.exp()

        self.action = self.sample_action()

        return self.action

    def calc_parameter_reg(self):
        """
        Original L2 penalty + distance-weighted recurrent penalty.
        """

        # original L2 regularisation on all parameters
        l2_loss = self.W_reg * torch.stack(
            [torch.square(p).sum() for p in self.parameters()]
        ).sum()

        # distance penalty on recurrent weights only
        dist_loss = self.dist_reg * (
            torch.abs(self.Wrec) * self.distance_matrix
        ).sum()

        return l2_loss + dist_loss


# corticallyembedded version of LineEmbeddedRNN
class CorticallyEmbeddedRNN(LineEmbeddedRNN):
    classname = "CorticallyEmbeddedRNN"
    label = "cortical_rnn"

    def __init__(
        self,
        env,
        Nrec=800,
        W_reg=1e-3,
        r_reg=1e-3,
        nonlin_output=False,
        dist_reg=1e-7,
        line_decay=0.12,
        line_init_scale=1.5,
        use_local_init=True,
        localize_loc_input=False,
        localize_rew_input=False,
        localize_wall_input=False,
        local_fraction=1.0 / 6.0,
        readout_mode="global",
        embedding_name="mpfc_very_focused",
        embedding_species="human",
        embedding_seed=42,
        anchor_area_names=None,
        **kwargs,
    ):
        self.embedding_name = embedding_name
        self.embedding_species = embedding_species
        self.embedding_seed = embedding_seed
        self.anchor_area_names = (
            ["p32", "p32pr", "d32"]
            if anchor_area_names is None
            else list(anchor_area_names)
        )

        super(CorticallyEmbeddedRNN, self).__init__(
            env,
            Nrec=Nrec,
            W_reg=W_reg,
            r_reg=r_reg,
            nonlin_output=nonlin_output,
            dist_reg=dist_reg,
            line_decay=line_decay,
            line_init_scale=line_init_scale,
            use_local_init=use_local_init,
            localize_loc_input=localize_loc_input,
            localize_rew_input=localize_rew_input,
            localize_wall_input=localize_wall_input,
            local_fraction=local_fraction,
            readout_mode=readout_mode,
            **kwargs,
        )

    @property
    def name(self):
        nonlin_str = "nonlinout" if self.nonlin_output else "linout"
        basename = super(VanillaRNN, self).name

        # if embedding has auto anchor-unit file, name the run accordingly
        if (self._embedding_dir() / "anchor_unit_indices.npy").exists():
            anchor_str = "autoanchor"
        else:
            anchor_str = "-".join(self.anchor_area_names)

        return (
            f"{basename}/N{self.Nrec}_{nonlin_str}"
            f"_cortical_{self.embedding_name}_eseed{self.embedding_seed}"
            f"_ld{self.line_decay}_dr{self.dist_reg}"
            f"_lfrac{self.local_fraction}"
            f"_loc{int(self.localize_loc_input)}"
            f"_rew{int(self.localize_rew_input)}"
            f"_wall{int(self.localize_wall_input)}"
            f"_ro{self.readout_mode}"
            f"_anchor{anchor_str}"
        )

    def _embedding_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        return (
            repo_root
            / "data"
            / "embedding"
            / "subsampled"
            / self.embedding_species
            / self.embedding_name
            / f"units={self.Nrec}_seed={self.embedding_seed}"
        )

    def _load_cortical_embedding(self):
        embedding_dir = self._embedding_dir()
        if not embedding_dir.exists():
            raise FileNotFoundError(
                f"Could not find cortical embedding at {embedding_dir}. "
                "Create it first with pysta.embedding.mpfc_embedding."
            )

        embedding = load_embedding(embedding_dir)

        D = torch.tensor(embedding["distance_matrix"], dtype=torch.float32)
        if D.shape != (self.Nrec, self.Nrec):
            raise ValueError(
                f"Embedding distance matrix has shape {tuple(D.shape)}, "
                f"expected {(self.Nrec, self.Nrec)}."
            )

        # keep distance scale numerically similar to line version
        D = D / D.mean().clamp(min=1e-8) 

        area_labels = list(embedding["area_labels"])
        sampled_indices = torch.tensor(embedding["sampled_indices"], dtype=torch.long)

        return D, area_labels, sampled_indices

    def _load_auto_anchor_unit_indices(self):
        # auto-load anchor/input zone from embedding dir if present
        anchor_path = self._embedding_dir() / "anchor_unit_indices.npy"

        if not anchor_path.exists():
            return None

        anchor_unit_indices = np.load(anchor_path).astype(np.int64).reshape(-1)
        anchor_unit_indices = np.unique(anchor_unit_indices).astype(np.int64)

        if len(anchor_unit_indices) == 0:
            raise ValueError(f"Auto anchor file is empty: {anchor_path}")

        if np.min(anchor_unit_indices) < 0 or np.max(anchor_unit_indices) >= self.Nrec:
            raise ValueError(
                f"Auto anchor indices in {anchor_path} are outside valid unit range "
                f"0..{self.Nrec - 1}."
            )

        print("\nUsing auto anchor/input zone from embedding:")
        print(f"  {anchor_path}")
        print(f"  anchor units: {len(anchor_unit_indices)} / {self.Nrec}")
        print(f"  fraction: {len(anchor_unit_indices) / float(self.Nrec):.4f}")

        return torch.tensor(anchor_unit_indices, dtype=torch.long)

    def _make_cortical_band_mask_from_anchor(self, anchor_idx):
        mask = torch.zeros(self.Nrec, dtype=torch.float32)
        k = self._local_band_size()
        nearest = torch.argsort(self.distance_matrix[anchor_idx])[:k]
        mask[nearest] = 1.0
        return mask

    def _make_cortical_band_mask_from_unit_indices(self, unit_indices):
        # exact mask from precomputed anchor/input units
        mask = torch.zeros(self.Nrec, dtype=torch.float32)
        unit_indices = torch.as_tensor(unit_indices, dtype=torch.long)
        mask[unit_indices] = 1.0
        return mask

    def _choose_anchor_index(self, area_labels):
        candidate_inds = [i for i, area in enumerate(area_labels) if area in self.anchor_area_names]
        if len(candidate_inds) == 0:
            raise ValueError(
                f"None of anchor_area_names={self.anchor_area_names} "
                "were present in sampled area labels."
            )

        cand = torch.tensor(candidate_inds, dtype=torch.long)

        # choose centroid-most sampled unit within proxy hippocampal-facing subset
        cand_D = self.distance_matrix[cand][:, cand]
        anchor_local = torch.argmin(cand_D.mean(dim=1))

        return int(cand[anchor_local].item())

    def _choose_anchor_index_from_unit_indices(self, unit_indices):
        # representative anchor for an explicit anchor/input zone
        cand = torch.as_tensor(unit_indices, dtype=torch.long)

        if len(cand) == 1:
            return int(cand[0].item())

        cand_D = self.distance_matrix[cand][:, cand]
        anchor_local = torch.argmin(cand_D.mean(dim=1))

        return int(cand[anchor_local].item())

    def initialise_weights(self):
        """
        Same logic as LineEmbeddedRNN, but replace the 1D line geometry
        with the saved cortical MPFC embedding.
        """

        # RNN initial condition
        self.z0 = nn.Parameter(torch.randn(self.Nrec, 1), requires_grad=True)

        # load cortical embedding
        D, area_labels, sampled_indices = self._load_cortical_embedding()
        self.sampled_area_labels = area_labels

        self.register_buffer("sampled_vertex_indices", sampled_indices)
        self.register_buffer("distance_matrix", D)

        # prefer auto anchor/input zone saved inside embedding dir
        auto_anchor_unit_indices = self._load_auto_anchor_unit_indices()

        if auto_anchor_unit_indices is not None:
            anchor_idx = self._choose_anchor_index_from_unit_indices(
                auto_anchor_unit_indices
            )

            same_end_mask = self._make_cortical_band_mask_from_unit_indices(
                auto_anchor_unit_indices
            )

            # opposite end = unit farthest on average from full input/anchor zone
            mean_distance_to_anchor_zone = self.distance_matrix[
                :, auto_anchor_unit_indices
            ].mean(dim=1)
            opposite_anchor_idx = int(torch.argmax(mean_distance_to_anchor_zone).item())

            self.anchor_unit_indices = [
                int(x) for x in auto_anchor_unit_indices.detach().cpu().numpy()
            ]

        else:
            # original behaviour: define anchor by sampled parcel labels
            anchor_idx = self._choose_anchor_index(area_labels)
            opposite_anchor_idx = int(torch.argmax(self.distance_matrix[anchor_idx]).item())

            same_end_mask = self._make_cortical_band_mask_from_anchor(anchor_idx)
            self.anchor_unit_indices = None

        self.anchor_index = anchor_idx
        self.opposite_anchor_index = opposite_anchor_idx

        self.register_buffer("same_end_unit_mask", same_end_mask)
        self.register_buffer(
            "opposite_end_unit_mask",
            self._make_cortical_band_mask_from_anchor(opposite_anchor_idx),
        )

        if self.readout_mode == "global":
            readout_mask = torch.ones(self.Nrec, dtype=torch.float32)
        elif self.readout_mode == "same_end":
            readout_mask = self.same_end_unit_mask.clone()
        elif self.readout_mode == "opposite_end":
            readout_mask = self.opposite_end_unit_mask.clone()
        else:
            raise ValueError(f"Unknown readout_mode: {self.readout_mode}")

        self.register_buffer("readout_unit_mask", readout_mask)

        # local recurrent init on cortical geodesic distances
        locality = torch.exp(-self.distance_matrix / self.line_decay)

        Wrec = (
            self.line_init_scale
            * torch.randn(self.Nrec, self.Nrec)
            / np.sqrt(self.Nrec)
        )
        if self.use_local_init:
            Wrec = Wrec * locality

        self.Wrec = nn.Parameter(Wrec, requires_grad=True)

        # input matrix is learnable; routing handled by masks
        self.Win = nn.Parameter(
            torch.randn(self.Nrec, self.Nin) / np.sqrt(self.Nin),
            requires_grad=True,
        )

        self.brec = nn.Parameter(torch.zeros(self.Nrec, 1), requires_grad=True)

        # re-use existing input routing
        self._build_input_masks()

        # output layer exactly as before
        if self.nonlin_output:
            self.Wout1 = nn.Parameter(
                torch.randn(int(np.round(self.Nrec / 2)), self.Nrec)
                / np.sqrt(self.Nrec),
                requires_grad=True,
            )
            self.bout1 = nn.Parameter(
                torch.zeros(self.Wout1.shape[0], 1),
                requires_grad=True,
            )

            self.Wout = nn.Parameter(
                torch.randn(self.Nout, self.Wout1.shape[0])
                / np.sqrt(self.Wout1.shape[0]),
                requires_grad=True,
            )
            self.bout = nn.Parameter(torch.zeros(self.Nout, 1), requires_grad=True)

        else:
            self.Wout = nn.Parameter(
                torch.randn(self.Nout, self.Nrec) / np.sqrt(self.Nrec),
                requires_grad=True,
            )
            self.bout = nn.Parameter(torch.zeros(self.Nout, 1), requires_grad=True)

        return

#%% Handcrafted spacetime attractor

class SpaceTimeAttractor(BaseAgent):
    classname = "SpaceTimeAttractor"
    label = "sta"
    
    def __init__(self, env, beta = 9.0, tau = 50, iters_per_action = 400, shift_time = 2.0, rec_noise = 1e-1, adj_noise = 1e-2, **kwargs):
        """
        Spacetime attractor that optimises a reward function

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        beta : Float
            Temperature parameter for the reward function. Performance is somewhat sensitive to this.
            If beta is too low, the network can converge to a diffuse representation instead of a clean trajectory to the goal.
            If beta is too high, the representation can 'teleport' for long trajectories.
        tau : Float
            Time constant for the dynamics
        shift_time : Float
            Number of time constants for which a feedforward component is included in the dynamics.
        rec_noise : Float
            Magnitude of the noise added to the recurrent dynamics.
        adj_noise : Float
            Magnitude of the noise added to each element of the recurrent weights
        """
        
        # set some hyperparameters
        self.num_modules = env.max_steps+1 # Number of subspaces; current loc plus max_steps future locs
        self.num_locs = env.num_locs
        self.batch = env.batch
        self.adj_noise = adj_noise
        self.shift_time = shift_time
        
        self.Nrec = self.num_modules * self.num_locs # total number of neurons across modules
        self.beta = beta # reward function temperature parameters
        self.exact = False # perform exact inference
        
        # initialise BaseAgent
        kwargs["tau"] = tau
        kwargs["iters_per_action"] = iters_per_action
        super(SpaceTimeAttractor, self).__init__(env, **kwargs)
        self.rec_noise = rec_noise # recurrent noise to ensure robustness
    
    def phi(self, x):
        """
        We use an exponential nonlinearity.
        """
        return x.exp()
    
    def initialise_weights(self):
        """
        Instantiate the handcrafted model parameters
        """

        A = self.env.adjacency.clone() # adjacency matrix of the environment
        adj_noise = self.adj_noise # how much noise will we add
        bias = -5e-3 # bias added to the noise
        
        # instantiate recurrent weight matrices
        self.Wrec_fwd, self.Wrec_bwd = [torch.zeros(self.batch, self.Nrec, self.Nrec) for _ in range(2)]
        self.Win = torch.zeros(self.Nrec, self.Nin) # input weights
        self.Wout = torch.zeros(self.env.num_locs, self.Nrec) # inherently allocentric; can convert to egocentric subsequently
        self.Wrec_shift = torch.zeros(self.Nrec, self.Nrec) # weights for the 'shift' dynamics
        
        # set recurrent weights
        for mod1 in range(self.num_modules-1): # for each subspace
            # indices of the weight matrices corresponding to this subspace and the next
            mod1_inds, mod2_inds = [torch.arange(self.num_locs)+self.num_locs*ind for ind in [mod1, mod1+1]]
            self.Wrec_shift[mod1_inds, mod2_inds] = torch.ones(self.num_locs) # shift connections from delta+1 to delta
            for i1, ind1 in enumerate(mod1_inds):
                for i2, ind2 in enumerate(mod2_inds):
                    # weights are just the adjacency matrix plus some noise
                    self.Wrec_fwd[:, ind2, ind1] = A[:, i2, i1].clone() + (torch.rand(self.batch) - 2.0)*adj_noise + bias
                    self.Wrec_bwd[:, ind1, ind2] = A[:, i1, i2].clone() + (torch.rand(self.batch) - 2.0)*adj_noise + bias
        
        # total recurrent weights combine forward and backward connections between subspaces
        self.Wrec = self.Wrec_fwd + self.Wrec_bwd
        
        # input weights
        self.Win[:self.num_locs, :self.num_locs] = torch.eye(self.num_locs) # current loc
        self.Win[self.num_locs:, 2*self.num_locs:-2*self.num_locs] = torch.eye(self.Win.shape[0] - self.num_locs) # future reward (zero for _current_ reward)
        
        # output is just the delta=1 subspace
        self.Wout[:, self.num_locs:2*self.num_locs] = torch.eye(self.num_locs)
        
        # initial condition is uniform in space at each time
        self.z0 = torch.log(torch.ones(self.num_modules, self.num_locs) / self.num_locs)
        
        #bias is zero for this agent 
        self.brec = torch.zeros(self.Nrec, 1)
        
        return

    def calc_policy(self):
        """compute policy from firing rate"""
        self.pi = (self.Wout @ self.r.reshape(self.batch, -1, 1))[..., 0] # policy ends up being activity in second module
        if self.env.output_format == "egocentric": # optionally convert to an egocentric policy
            self.pi = self.allo_to_ego_pi(self.pi)

    def obs_to_rew_func(self, observation, flat = False):
        """This function extract the location and reward information from the inputs and puts it into the right format"""
        obs_mod = observation.reshape(self.batch, -1, self.num_locs) # batch by input type by location within module
        rew_inds = torch.cat([torch.arange(1), torch.arange(2, obs_mod.shape[1]-2)]) # indicies of the inputs corresponding to current location and future reward
        logrews = obs_mod[:, rew_inds, :] # reward function (batch, modules, locs)
        logrews[:, 0, :] *= 20.0 # strong signal to start at current location
        logrews *= self.beta # multiply by temperature parameter
        logrews -= logrews.logsumexp(axis = -1, keepdims = True) # turn into a distribution over locations and add final vector dimension 
        if flat: # flatten back into a vector instead of a matrix
            obs_mod[:, rew_inds, :] = logrews # replace reward function in observation
            return obs_mod.reshape(self.batch, -1, 1)
        else:
            return logrews[..., None]

    def step(self, observation):

        clip_minval = torch.tensor(-100) # threshold value for inhbition

        # first extract reward function from observation
        logrews = self.obs_to_rew_func(observation.clone(), flat = True) # flatten for each trial (batch, modules*locs)

        # select how many network iterations to run
        network_iters = self.iters_per_action if type(self.iters_per_action) in [int, np.int32, np.int64] else np.random.choice(self.iters_per_action)
        
        for iter_ in range(network_iters): # for each iteration

            rs = self.r.reshape(self.batch, -1, 1) # firing rate but add vector dimension
            # compute change in activity
            dzdt = (self.Win @ logrews + torch.maximum(clip_minval.exp(), self.Wrec_fwd @ rs).log() + torch.maximum(clip_minval.exp(), self.Wrec_bwd @ rs).log())

            # optionally add a 'shift component' to the dynamics for a period after each action
            if (iter_ < self.tau*self.shift_time) and (self.env.step_num > 0):
                dzdt += torch.maximum(clip_minval.exp(), self.Wrec_shift @ rs).log()
            
            # reshape update and bias
            dzdt = dzdt.reshape(self.batch, -1, self.num_locs)
            bias = self.brec.reshape(1, -1, self.num_locs)

            # sample recurrent noise
            rec_noise = torch.randn(dzdt.shape, device = self.z0.device) * self.rec_noise

            # update activity
            self.z = (1-1/self.tau) * self.z + 1/self.tau * (dzdt + rec_noise + bias)
            
            # normalize and threshold
            self.z = self.z - self.z.logsumexp(axis = -1, keepdims = True) # normalise
            self.z = torch.clip(self.z, min = clip_minval, max = 0.0) # threshold
            self.r = self.phi(self.z) # apply exponential nonlinearity to compute firing rates

            if self.store_all_activity: # optionally store activity at every RNN iteration
                self.all_acts[0].append(self.r.detach().numpy())
                self.all_acts[1].append( self.env.loc.detach().numpy())
                self.all_acts[2].append(self.env.step_num)

        #compute the policy
        self.calc_policy()
        # sample an action from the policy
        self.action = self.sample_action()
        
        return self.action
            
    def reset(self):
        """Reset the agent. This involves the BaseAgent reset and then re-initialising weights in case the environment changed"""
        super(SpaceTimeAttractor, self).reset()
        self.initialise_weights() # reinitialize weights
        
        return

    def plot_representation(self, filename = None, trial_num = 0, **kwargs):
        """
        Generate a summaruy plot for a trial

        Parameters
        ----------
        filename : str
            filename to save to
        trial_num : int
            which trial number within the batch to plot
        """

        # find out which time points have not finished
        fig, axs = plt.subplots(1, self.num_modules, figsize = (2*self.num_modules, 2))
        for t, ax in enumerate(axs): # for each time point, plot a panel
            loc = self.env.loc[trial_num] if t == 0 else torch.argmax(self.r[trial_num, t]) # predicted location at this time
            self.env.plot(loc = loc, vmap = self.r[trial_num, t, :], step_num = t, ax = ax, trial_num = trial_num,
                          cmap  = "YlOrRd", plot_optimal_actions = t<len(axs)-1, vmin = -0.1, vmax = 1.2, **kwargs)
        plt.tight_layout = True
        if filename is not None:
            plt.savefig(filename, bbox_inches = "tight")
        plt.close()
        
        return

    
#%% Successor representation agent
class SRLearner(BaseAgent):
    classname = "SRLearner"
    label = "sr"

    def __init__(self, env, gamma = 0.95, beta = 5.0, **kwargs):
        """
        Successor representation agent that navigates in a maze

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        gamma : Float
            The discount factor used to compute the successor matrix
        beta : Float
            The temperature parameter used to compute the policy
        """
        
        # instantiate class and save some parameters
        self.gamma = gamma
        self.beta = beta # temperature parameter
        super(SRLearner, self).__init__(env, **kwargs)
        self.Nrec = 2*self.env.num_locs # expected occupancy of each location and expected value of each location
        
        return
    
    def phi(self, x):
        return x # no nonlinearity
    
    def reset(self):
        """Reset Baseagent then recompute successor matrix"""
        super(SRLearner, self).reset()
        adj = self.env.adjacency # (batch, to, from)
        T = adj / adj.sum(1, keepdims = True) # transition matrix
        self.M = torch.linalg.inv(torch.eye(T.shape[-1])[None, ...] - self.gamma*T) # analytically compute successor matrix
        self.r = None # no neural activity
        
        return
    
    def initialise_weights(self):
        """
        Model parameters are empty; we will compute everything analytically
        """
        self.z0 = torch.zeros(self.env.num_locs) # z is value, which we initialize to zero
        return

    def step(self, observation):
        """
        Perform one 'update step'
        For simplicity, we just multiply the average occupancy with the average reward function.
        If the reward format is 'relative', this ends up being the 'reward-to-go'.
        In the future, we could think about also using the 'occupancy-to-go' instead of just exponentially decayed occupancy.

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """

        rew_func = observation.reshape(self.env.batch, -1, self.env.num_locs)[:, 1:-2, :].clone() # reward function (batch, steps, locs)

        # compute average reward
        mean_rew = rew_func[:, 1:, :].mean(1, keepdims = True) # ignore reward at current time
        
        # compute value function
        self.values = (mean_rew @ self.M)[:, 0, :] # (batch, locs)
        self.z = self.values # neural potential is just the value function
        exp_occ = self.M[self.env.batch_inds, :, self.env.loc] # expected occupancy
        # 'neural activity' is values and occupancy. This is not used for any computation, but may be interesting for decoding analyses.
        self.r = torch.cat([self.values, exp_occ], 1)[..., None]
        
        # compute policy
        self.pi = (self.beta * self.values).exp() # allocentric policy
        self.pi /= self.pi.sum(-1, keepdims = True) # normalize
        
        if self.env.output_format == "egocentric":
            self.pi = self.allo_to_ego_pi(self.pi) # optionally convert to egocentric policy

        self.action = self.sample_action() # sample an action from the policy
        
        return self.action
    
    def plot_trial_values(self, filename = None, run_trial = True, trial_num = 0, cmap = "coolwarm"):
        """Function for plotting the value function for a trial"""
        
        if run_trial: # optionally simulate a new trial
            self.forward(store = True)
        vmap = [s["zs"][trial_num].flatten() for s in self.store]
            
        self.plot_trial(filename = filename, run_trial = False, trial_num = trial_num, vmap = vmap, cmap = cmap)

#%% TD learning agent

class TDLearner(BaseAgent):
    classname = "TDLearner"
    label = "td"

    def __init__(self, env, beta = 5.0, tau = 20, **kwargs):
        """
        Temporal difference learner that navigates in a maze

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        beta : Float
            The temperature parameter used to compute a policy
        tau : Float
            The inverse of the learning rate
        """
        
        # instantiate super class
        self.beta = beta # temperature parameter
        kwargs["tau"] = tau
        super(TDLearner, self).__init__(env, **kwargs)
        
        return
    
    def phi(self, x):
        return x # no nonlinearity
    
    def reset(self):
        """Reset BaseAgent and store initial location"""
        super(TDLearner, self).reset()
        self.prev_loc = self.env.loc
        self.prev_not_finished = torch.ones(self.env.num_locs)
        return
    
    def initialise_weights(self):
        """
        Model parameters are the values
        """
        self.values = torch.zeros(self.env.num_locs)+1.0 # optimistic value initialisation
        self.z0 = self.values.clone() # our firing rates will also be the values
        return
    
    def run_td_update(self):
        """Run a TD update on the value function"""
        
        new_rew = self.env.latest_rew # the reward we just got
        prev_V = self.values[self.prev_loc] # value of previous location
        new_V = self.values[self.env.loc] * (~self.env.finished).to(float) # value of new location (no more value for finished trials)
        
        self.td_error = (new_rew + new_V - prev_V) * self.prev_not_finished  # TD error (set to zero for finished episodes)
        
        # previous locations
        prev_locs_1hot = F.one_hot(self.prev_loc, num_classes = self.env.num_locs) # (batch, num_locs)
        # batched TD update
        updates = (prev_locs_1hot * self.td_error[:, None]).sum(0) / self.prev_not_finished.sum() # avg over all td errors for each location in the batch
        self.values += (1/self.tau)*updates # update values

    def step(self, observation):
        """
        Perform one 'update step'
        This involves both choosing an action and updating our value function

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """

        # first select an action
        self.pi = (self.beta * self.values).exp()[None, ...] + torch.zeros(observation.shape[0], self.env.num_locs) # # allocentric policy is proportional to exponential value
        self.pi /= self.pi.sum(-1, keepdims = True) # normalize
        if self.env.output_format == "egocentric":
            self.pi = self.allo_to_ego_pi(self.pi) # optionally convert to egocentric

        self.action = self.sample_action() # sample an action from the policy
        
        # now update our values based on previous experience
        if self.env.step_num >= 1: # can only run a TD update once we start collecting experience
            self.run_td_update()
            
        self.z = self.values[None, :] + torch.zeros(self.env.batch, self.env.num_locs) # potential is just value
        self.r = self.phi(self.z) # firing rate is just value (this is not used for computation, just for visualisation)
        self.prev_loc = self.env.loc # save previous location
        self.prev_not_finished = (~self.env.finished).to(float) # which trials are done?
        
        return self.action
    
    def forward(self, store = False):
        """
        Run a single batch of trials

        Parameters
        ----------
        store : bool
            if true, store environment and agent states after every action
            
        Returns
        ----------
        avg_loss : tensor
            average loss across trials in a batch
        """
        
        loss = super(TDLearner, self).forward(store = store) # run default forward pass
        self.run_td_update() # important to update values based on final sample as well
        return loss
    
    def plot_trial_values(self, filename = None, run_trial = True, trial_num = 0, cmap = "YlOrRd"):
        """Function for plotting value function after a batch of trials"""
        if run_trial: # optionally simulate a new trial
            self.forward(store = True)
        vmap = [s["zs"][trial_num].flatten() for s in self.store]
            
        self.plot_trial(filename = filename, run_trial = False, trial_num = trial_num, vmap = vmap, cmap = cmap)



#%% Dynamic programming agent (value agent in space and time)

class DPAgent(BaseAgent):
    classname = "DPAgent"
    label = "dp"
    
    def __init__(self, env, beta = 5.0, **kwargs):
        """
        Dynamic programming agent that just computes an optimal spacetime value function once and for all

        Parameters
        ----------
        env : MazeEnv
            Environment that the agent is going to interact with
        beta : Float
            Temperature parameter for computing a policy
        """
        
        # store parameters
        self.beta = beta
        self.Nrec = env.num_locs + env.max_steps + np.prod(env.vs[0].shape) # location, time-in-trial, and spacetime value function
        
        # initialise BaseAgent
        super(DPAgent, self).__init__(env, **kwargs)
        
        return
    
    def phi(self, x):
        return x # no nonlinearity
    
    def initialise_weights(self):
        self.z0 = torch.zeros(self.Nrec, 1) # no parameters
    
    def update_z_and_r(self):
        """This just involves concatenating the relevant environment variables into a vector that will be used for decoding analyses"""
        self.t = torch.tensor(max(0, self.env.step_num)) # current time (everything during planning is called '0')
        self.flat_t = F.one_hot(self.t, num_classes = self.env.max_steps)+torch.zeros(self.env.batch, self.env.max_steps) # one-hot version
        self.loc = F.one_hot(self.env.loc, num_classes = self.env.num_locs) # current location as a one-hot
        
        self.values = self.env.vs # value function
        # set potential and firing rate to the concatenation of time, location, and values
        self.z = torch.cat([self.values.reshape(self.env.batch, -1), self.loc, self.flat_t], dim = -1)[..., None]
        self.r = self.z.clone()
        
    def reset(self):
        """Run BaseAgent reset step, and then re-cache the value function and recompute representation since the environment may have changed"""
        super(DPAgent, self).reset()
        self.values = self.env.vs
        self.update_z_and_r()
        
        return
    
    def step(self, observation):
        """
        Perform one 'update step'
        This involves updating the neural representation and sampling an action

        Parameters
        ----------
        observation : tensor
            The observation at this point in time

        Returns
        ----------
        action : tensor
            action taken for each trial in the batch
        """

        # first update representations
        self.update_z_and_r()
        
        # then compute policy
        next_values = self.values[:, self.t+1, :] # values at every upcoming point in spacetime
        
        self.pi = (self.beta * next_values).exp() # allocentric policy is proportional to exponential value
        self.pi /= self.pi.sum(-1, keepdims = True) # normalize
        
        if self.env.output_format == "egocentric":
            self.pi = self.allo_to_ego_pi(self.pi) # optionally convert to egocentric

        # and sample an action
        self.action = self.sample_action() # sample an action from the policy
        
        return self.action
