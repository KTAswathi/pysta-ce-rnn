
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from .plot_utils import plot_flat_frame
from .maze_utils import sample_maze, compute_adjacency, compute_shortest_dists

class MazeEnv():
    
    def __init__(self, side_length: int = 4, max_steps: int = 6, changing_trial_maze: bool = False, dynamic_rew: bool = True, changing_trial_rew: bool = True, batch_size: int = 11, sample_wall_num: int = 8,
                 rew_landscape: bool = True, relative_rew: bool = True, output_format: str = "allocentric", planning_steps = [1,], working_memory = False, inp_noise = 0.0, inp_noise_planning = None, 
                 rew_goal = 0.6, rew_nogoal = -0.6, **kwargs):
        """
        Maze environment with a variety of possible configurations and reward functions

        Parameters
        ----------
        side_length : int
            Side length of the square maze
        max_steps : int
            Maximum number of actions an agent can take in the environment (exact number for non-absorbing reward functions)
        changing_trial_maze : bool
            Whether the maze is fixed (False) or changes every trial (True)
        dynamic_rew : bool
            Whether the reward function is constant at each time step (False) or changes between timesteps (True)
        changing_trial_rew : bool
            Whether the reward function is fixed (False) or changes (True) every trial
        batch_size : int
            Batch size used in the environment
        sample_wall_num : int
            The maximum numer of unique wall configurations sampled in a given batch. If sample_wall_num < batch, some trials in each batch will use the same maze
        rew_landscape : bool
            Whether to use a 'reward landscape' with different values at every location (True) or a single absorbing goal (False)
        relative_rew : bool
            Whether the reward input is given relative to the current time (True) or as a constant input (False)
        output_format : str
            The output format can either be egocentric actions ("egocentric") or allocentric desired locations ("allocentric")
        planning_steps : list of ints
            Number of environment steps in the 'planning period'
        working_memory : bool
            If true, the agent observes the reward function only during the planning period
        inp_noise : float
            Fraction of noise added to the input (inp = (1-frac)*true_inp + frac*U(0,1))
        inp_noise_planning : None or float
            If None, the same amount of input noise is added at the 'planning' and 'execution' stages.
            If float, this is the amount of input noise used during the 'planning' stage.
        """
        
        # assert that we have a possible environment
        assert output_format in ["egocentric", "allocentric"]
        assert side_length >= 1
        assert max_steps >= 1
        assert batch_size >= 1
        assert np.amin(planning_steps) >= 0
        assert (sample_wall_num is None) or (sample_wall_num >= 1)
        
        # store some hyperparameters
        self.side_length = side_length
        self.num_locs = side_length**2 # total number of locations is the square of the arena side length
        self.max_steps = max_steps
        self.batch = batch_size
        self.batch_inds = torch.arange(self.batch) # useful to catch a tensor that indexes each element in a batch
        self.sample_wall_num = sample_wall_num
        self.rew_landscape = rew_landscape
        self.relative_rew = relative_rew
        self.num_actions = 5 # number of possible egocentric actions
        self.planning_steps = planning_steps
        self.working_memory = working_memory
        self.rew_goal, self.rew_nogoal = rew_goal, rew_nogoal
        
        self.output_format = output_format
        # dimensionality of the output space is either the number of actions or number of locs
        self.output_dim = self.num_actions if output_format == "egocentric" else self.num_locs
        
        self.changing_trial_maze = changing_trial_maze
        self.changing_trial_rew = changing_trial_rew
        self.dynamic_rew = dynamic_rew

        # now compute the dimensionality of the observations
        self.obs_dim = self.num_locs # where am I
        self.obs_dim += (self.max_steps+1)*self.num_locs # reward function at each point in time
        self.obs_dim += self.num_locs * 2 # binary input indicating whether there is a wall or not at each edge
        
        # noise parameters
        self.inp_noise = inp_noise
        self.inp_noise_planning = (inp_noise if inp_noise_planning is None else inp_noise_planning)
        
        # precompute for a base environment with no walls (i) the transition matrix and (ii) the the loc reached from every (loc, action) combination
        base_walls = torch.zeros(self.num_locs, self.num_actions) # base environment with no walls
        self.nowall_neighbors = compute_adjacency(base_walls)[1]
        
        # initialize the environment
        self.sample_maze() # sample maze configuration(s)
        self.sample_reward() # sample reward function(s)
        self.reset() # initialize everything else
        
        return

    @property
    def name(self):
        """generates a string representation of the environment"""
        
        if self.working_memory:
            rew_obs_str = "planrew"
        elif self.relative_rew:
            rew_obs_str = "relrew"
        else:
            rew_obs_str = "constrew"
            
        out_str = "allo" if self.output_format == "allocentric" else "ego"
        dynamic_str = "dynamic-rew" if self.dynamic_rew else "static-rew"
        changing_maze_str = "changing-maze" if self.changing_trial_maze else "constant-maze"
        changing_rew_str = "changing-rew" if self.changing_trial_rew else "constant-rew"
        rewtype_str = "landscape" if self.rew_landscape else "goal"
        L_str = f"L{self.side_length}"
        if type(self.planning_steps) in [int, np.int32, np.int64]:
            plan_step_str = f"plan{self.planning_steps}"
        else:
            plan_step_str = f"plan{'-'.join([str(val) for val in self.planning_steps])}"
        
        return f"MazeEnv_{L_str}_max{self.max_steps}/{rewtype_str}_{changing_rew_str}_{dynamic_str}_{changing_maze_str}/{out_str}_{rew_obs_str}_{plan_step_str}"

    def sample_maze(self, walls = None):
        """
        function for sampling a new batch of maze configurations and computing various derived metrics
        
        Parameters
        ----------
        walls : tensor
            optionally provide walls instead of resampling
        """
        
        # self.walls: (batch, N, Nactions)
        if walls is not None:
            self.walls = walls
        elif (not self.changing_trial_maze):
            # if it is a fixed maze environment, we just sample one maze and broadcast to the full batch
            self.walls = torch.tensor(sample_maze(self.side_length))[None, ...] + torch.zeros(self.batch, 1, 1)
        elif self.sample_wall_num is None:
            # if it is a changing environment, sample a different maze for each trial in the batch by default
            self.walls = torch.tensor(np.array([sample_maze(self.side_length) for _ in self.batch_inds]), dtype = torch.float32)
        else:
            # to speed things up, optionally just sample 'self.sample_wall_num' different configurations
            # we will broadcast these to the full batch size later
            self.walls = torch.tensor(np.array([sample_maze(self.side_length) for _ in range(self.sample_wall_num)]), dtype = torch.float32)
        
        # get an array of the next locs given any (loc, action) combination in each maze
        num_mazes, maze_inds = self.walls.shape[0], torch.arange(self.walls.shape[0])
        self.neighbors = torch.cat([self.walls*torch.arange(self.num_locs)[None, :, None] + (1.0-self.walls)*self.nowall_neighbors[None, :, :-1], torch.arange(self.num_locs)[None, :, None]+torch.zeros(num_mazes, self.num_locs, 1)], -1).to(int)
        
        # get the adjacency matrix for each maze
        self.adjacency = torch.zeros(num_mazes, self.num_locs, self.num_locs)
        for s in range(self.num_locs): # for each loc
            for a in range(self.num_actions): # for each action
                self.adjacency[maze_inds, s, self.neighbors[:, s, a]] = 1 # the loc s' reached by taking a from s is adjacent

        # if we did not sample enough mazes for the full batch, broadcast them back to a full batch of trials
        if (self.changing_trial_maze) and (self.sample_wall_num is not None):
            sample_inds = self.batch_inds%self.sample_wall_num # maze index for each trial
            self.walls, self.neighbors, self.adjacency = self.walls[sample_inds, ...], self.neighbors[sample_inds, ...], self.adjacency[sample_inds, ...]

    def compute_value_function(self):
        """
        computes the value function corresponding to the current reward function
        """
        # now we want to compute the optimal value function at every point in spacetime
        self.neighbor_mask = (self.adjacency-1) # 0 for adjacent locations, -1 for non-adjacent
        self.neighbor_mask[self.neighbor_mask < -0.5] = -torch.inf # 0 and -inf
        self.vs = torch.zeros(self.batch, self.max_steps+1, self.num_locs) # initialize value function
        self.vs[:, -1, :] = self.rews[:, -1, :] # best state at the end is just the one with the highest instantaneous reward

        for t in range(self.max_steps): # now propagate values back using Bellman equation
            # compute v(s, t) = max_{s_t' \in s_t:T s.t. s_t = s} \sum_t'=t^T r(s_t', t') = r(s, t) + max_{s' \in N(s)} v(s', t+1)

            # value of best location I can go from each prior location
            self.vs[self.batch_inds, self.max_steps-t-1, :] = torch.amax(self.vs[:, self.max_steps-t, None, :] + self.neighbor_mask, -1)
            
            # and the reward I'll get at that location
            self.vs[self.batch_inds, self.max_steps-t-1, :] += self.rews[:, self.max_steps-t-1, :]
        
    def sample_reward_landscape(self):
        """
        function that samples a new reward landscape for each trial and computes the corresponding value functions
        """
        
        # first we sample the actual rewards iid from U(-1, +1)
        if self.dynamic_rew:
            self.rews = torch.rand(self.batch, self.max_steps+1, self.num_locs)*2-1 # different reward functions for each trial
        else:
            self.rews = (torch.rand(self.batch, 1, self.num_locs)*2-1)+torch.zeros(self.batch, self.max_steps+1, self.num_locs) # just one reward function for all timepoints and broadcast
        
        if not self.changing_trial_rew: # reward function is the same across trials
            self.rews[...] = self.rews[:1, ...]

    def sample_path(self, length, turn_bias = 1e-3):
        """
        this function samples a goal trajectory that (i) never stays in the same location twice in a row,
        and (ii) only turns back on itself with low probability.

        Parameters
        ----------
        length : int
            length of the sampled trajectory
        reverse_prob : float
            probability of reversing (relative to the probability of visiting any one alternative loc)
        """

        path = torch.zeros(self.batch, length, dtype = int) # instantiate path
        path[:, 0] = torch.tensor(np.random.choice(self.num_locs, self.batch, replace = True)) # start somewhere random

        # we construct a new adjacency matrix that does not allow staying in the same place
        nostay_adjacency = self.adjacency.clone()
        nostay_adjacency[:, torch.arange(self.num_locs), torch.arange(self.num_locs)] = 0.0

        for i in range(length-1): # take a bunch of steps on the path
            ps = nostay_adjacency[self.batch_inds, path[:, i]] # adjacent locations
            if i >= 1: # if it's not the first step
                ps[self.batch_inds, path[:, i-1]] = turn_bias # small probability of turning back
            
            # sample the next location for every path
            path[:, i+1:i+2] = torch.multinomial(ps, 1)
            
        return path

    def sample_reward(self):
        """
        function that samples a new reward function for the environment.
        """

        # first consider the
        if self.rew_landscape:
            self.sample_reward_landscape()
        
        else:
            if self.dynamic_rew:
                # sample a full goal trajectory
                self.goal = self.sample_path(self.max_steps+1) #(batch, max_steps)
            else:
                # the goal remains in the same location throughout
                self.goal = torch.multinomial(torch.ones(self.batch, self.num_locs), 1) + torch.zeros(self.batch, self.max_steps+1, dtype = int) # (batch, max_steps)

            if not self.changing_trial_rew: # goal trajectory is the same across trials
                self.goal[...] = self.goal[:1, ...]
            
            self.rews = torch.zeros(self.batch, self.max_steps+1, self.num_locs) + self.rew_nogoal # default negative reward
            for t in range(self.max_steps+1):
                self.rews[self.batch_inds, t, self.goal[:, t]] = self.rew_goal # positive reward at the goal location
        
        self.compute_value_function()
    
    def reset(self, hard = False):
        """
        reset the environment.
        only resamples a maze and reward function if these change across trials
        resamples an initial location.
        
        Parameters
        ----------
        hard : bool
            if true, a new maze and initial location always resampled
        """

        if hard or self.changing_trial_maze: # resample a new maze if relevant
            self.sample_maze()
        if hard or self.changing_trial_rew: # resample new reward function if relevant
            self.sample_reward()

        pstarts = torch.ones(self.batch, self.num_locs)/(self.num_locs) # by default we can start anywhere
        if not self.rew_landscape: # but cannot start at the initial goal location
            pstarts[self.batch_inds, self.goal[:, 0]] = 0.0
        # sample new start locations
        self.loc = torch.multinomial(pstarts, 1)[:, 0] # (batch, )

        self.sample_rews = torch.zeros(self.batch, self.max_steps+1) # reward accumulated in this batch
        self.step_num = -self.planning_steps if type(self.planning_steps) in [int, np.int32, np.int64] else -np.random.choice(self.planning_steps) # action number. negative indicates planning phase.
        self.finished = torch.tensor([False for _ in self.batch_inds]) # which trials in the batch have finished
        self.latest_rew = self.update_empirical_reward()
        
        return
    
    def current_reward(self):
        """return reward at current location. This is 0 for episodes that are finshed."""
        rews = self.rews[self.batch_inds, max(0, self.step_num), self.loc]
        rews[self.finished] = 0.0
        return rews
    
    def update_empirical_reward(self):
        """update reward history"""
        curr_rew = self.current_reward()
        self.sample_rews[:, max(0, self.step_num)] = curr_rew
        return curr_rew

    def step(self, action):
        """
        this function updates the environment given a batch of actions

        Parameters
        ----------
        action : torch.tensor of ints
            index of the action taken for each trial.
        """
        
        if self.step_num < 0: # planning phase; just increment time and return location
            self.step_num += 1
            return self.loc
        
        self.step_num += 1 # increment action counter
        
        # now update our location
        if self.output_format == "egocentric":
            # the loc that our egocentric action takes us to
            self.loc = self.neighbors[self.batch_inds, self.loc, action]
        elif self.output_format == "allocentric":
            # the allocentric action directly corresponds to the next loc
            self.loc = action
        
        self.latest_rew = self.update_empirical_reward() # update empirical reward
        
        if (self.step_num >= self.max_steps):
            self.finished[:] = True # all trials are finished by definition after self.max_steps
        elif not self.rew_landscape:
            # trials where we reach the goal are finished in absorbing goal settings
            self.finished[self.loc == self.goal[:, self.step_num]] = True
            
        return self.latest_rew
    
    def optimal_actions(self, loc = None, step_num = None):
        """
        this function returns the optimal policy as a binary mask over actions.
        it returns the policy for the current environment state by default, but can also compute optimal policies
        for some query state via the 'loc' and 'step_num' kwargs.

        Parameters
        ----------
        action : torch.tensor of ints
            index of the action taken for each trial.

        Returns
        ----------
        optimal_actions : torch.tensor of floats
            binary array with 1 for optimal actions and 0 for other actions (batch, self.output_dim)
        """
        
        loc = self.loc if loc is None else loc # optionally get optimal policy for some query loc
        step_num = self.step_num if step_num is None else step_num # optionally get optimal policy for some query time
        step_num = max(0, step_num) # in the planning phase (step_num < 0), get optimal policy for first action
        
        # we need to find the actions with optimal long-term value
        new_locs = self.neighbors[self.batch_inds, loc, :] # for each action, where do I end up
        
        vnext = torch.zeros(self.batch, self.output_dim) - torch.inf # value of each action (default to -inf for impossible actions)
        for a in range(self.num_actions): # for each action
            new_locs_a = new_locs[:, a] # where would I end up
            new_vs = self.vs[self.batch_inds, step_num+1, new_locs_a] # what would the value be here
            if self.output_format == "egocentric":
                vnext[:, a] = new_vs # value at next loc for this action
            elif self.output_format == "allocentric":
                vnext[self.batch_inds, new_locs_a] = new_vs # the action is the next loc
        
        # binary tensor with 1 for optimal actions and 0 for other actions
        return torch.isclose(vnext, torch.amax(vnext, -1, keepdims = True)).to(torch.float32)

    def obs_inds(self):
        """
        indices in the input array that correspond to each different type of information
        """
        inds = {"loc": np.arange(self.num_locs)}
        inds["goal"] = np.arange((self.max_steps+1)*self.num_locs)+ self.num_locs
        inds["walls"] = np.arange(self.num_locs*2)+self.num_locs*(self.max_steps+2)
        return inds
    
    def observation(self):
        """
        generate observation for the current state of the environment
        """
        
        step_num = max(0, self.step_num) # planning phase observation is the same as initial observation
        
        # instantiate observation array
        obs = torch.zeros(self.batch, 4+self.max_steps, self.num_locs)
        obs[self.batch_inds, 0, self.loc] = 1. # where am I?
        
        # reward function
        if (not self.working_memory) or (self.step_num < 0): # optionally only rew info at planning time
            if self.relative_rew:
                obs[:, 1:self.max_steps+2-step_num, :] = self.rews[:, step_num:, :] # reward input shifts in time to always reflect reward relative to now
            else:
                obs[:, 1:self.max_steps+2, :] = self.rews # constant reward input
        
        # transition function
        obs[:, self.max_steps+2, :] = self.walls[:, :, 0] # walls to the right
        obs[:, self.max_steps+3, :] = self.walls[:, :, 2] # walls above
        obs = obs.flatten(-2, -1)
        
        # input noise is either 'inp_noise_planning' during planning, otherwise 'inp_noise'
        inp_noise_scale = self.inp_noise_planning if self.step_num < 0 else self.inp_noise
        inp_noise = torch.rand(self.batch, self.obs_dim) # sample some noise iid from U(0, 1)

        # combine observation and noise        
        noisy_obs = (1.0-inp_noise_scale)*obs  + inp_noise_scale * inp_noise
        
        return noisy_obs

    def plot(self, filename = None, trial_num = 0, goal = None, loc = None, step_num = None, optimal_actions = None, values = True, vmap = None, plot_optimal_actions = True, **kwargs):
        """
        function for plotting the current state of the environment
        """
        
        step_num = max(0, (self.step_num if step_num is None else step_num)) # planning phase is the same as first step
        
        if vmap is None: # default to either reward or value functions
            if values:
                vmap = self.vs[trial_num, step_num+1, :]
            else:
                vmap = self.rews[trial_num, step_num+1, :]
                
        if (goal is None) and (not self.rew_landscape): # plot goal location
            goal = self.goal[trial_num] if self.dynamic_rew else self.goal[trial_num, 0].item() # either the full sequence or just the constant goal location
        loc = self.loc[trial_num] if loc is None else loc
        
        if plot_optimal_actions and (optimal_actions is None):
            optimal_actions = self.optimal_actions(loc*np.ones(self.batch, dtype = int), step_num)[trial_num, :] # for this location
            
        plot_flat_frame(self.walls[trial_num], filename = filename, goal = goal, loc = loc, optimal_actions = optimal_actions, vmap = vmap, goal_step_num = step_num, **kwargs)
        
