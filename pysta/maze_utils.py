import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra

action_deltas = [np.array(delta) for delta in [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]]

def neighbors(cell, msize):
    dirs = [np.array(vec) for vec in [[1, 0], [-1, 0], [0, 1], [0, -1]]]
    Ns = [cell + dirs[a] for a in range(4)]
    minN, maxN = [np.array([f(N) for N in Ns]) for f in [np.amin, np.amax]]
    as_ = np.where((minN > -0.5) & (maxN < msize-0.5) )[0].astype(int)
    Ns = [Ns[a] for a in as_]
    return Ns, as_

def walk(maz, nxtcell, msize, visited = []):
    dir_map = {0 : 1, 1: 0, 2: 3, 3: 2}
    visited.append(nxtcell[0] * msize + nxtcell[1]) #add to list of visited cells
    neighs, as_ = neighbors(nxtcell, msize) # get list of neighbors

    for nnum in np.random.choice(len(neighs), len(neighs), replace = False): #for each neighbor in randomly shuffled list
        neigh, a = neighs[nnum], as_[nnum] # corresponding state and action
        ind = neigh[0] * msize + neigh[1]
        if ind not in visited: #check that we haven't been there
            maz[nxtcell[0], nxtcell[1], a] = 0.0 #remove wall
            maz[neigh[0], neigh[1], dir_map[a]] = 0.0 #remove reverse wall
            maz, visited = walk(maz, neigh, msize, visited = visited) #
    return maz, visited

def sample_maze(msize, holes = None):
    dirs = [np.array(vec) for vec in [[1, 0], [-1, 0], [0, 1], [0, -1]]]
    dir_map = {0 : 1, 1: 0, 2: 3, 3: 2}
    maz = np.ones((msize, msize, 4)) #start with walls everywhere
    cell = np.random.choice(msize, 2) #where do we start?
    maz, visited = walk(maz, cell, msize, visited = []) #walk through maze

    # remove a couple of additional walls to increase degeneracy
    if holes is None:
        holes = int(3 * (msize - 3)) #4 for Larena=4, 8 for Larena = 5
    # add permanent walls
    maz[msize-1, :, 0] = 0.5
    maz[0, :, 1] = 0.5
    maz[:, msize-1, 2] = 0.5
    maz[:, 0, 3] = 0.5
    for _ in range(holes):
        walls = np.array(np.where(maz == 1))
        wall = walls[:, np.random.choice(walls.shape[1])]
        cell, a = [wall[0], wall[1]], wall[2]

        neigh = np.array([cell[0], cell[1]]) + dirs[a] # what is the neighboring cell
        maz[cell[0], cell[1], a] = 0.0 #remove wall
        maz[neigh[0], neigh[1], dir_map[a]] = 0.0 #remove reverse wall

    maz[maz == 0.5] = 1.0 # reinstate permanent walls
    maz = np.array([maz[..., a].flatten() for a in range(4)]).T

    return maz

def loc_to_index(loc, L):
    """from loc in R^2 to index in R^L^2"""
    return loc[0]*L + loc[1]

def index_to_loc(index, L):
    """from index in R^L^2 to loc in R^2"""
    return np.array([index // L, index % L])

def compute_shortest_dists(transitions):
    """Run Djikstra's algorithm on transition matrix to compute shortest distances between pairs of states"""
    all_dists = dijkstra(transitions, directed=False, unweighted = True)
    return all_dists

def ind_a_ind(index, action, L):
    """
    compute the change in state index from taking an action
    """
    loc = index_to_loc(index, L) + action_deltas[action] # update location
    loc = np.minimum(np.maximum(loc, 0), L-1) # project back into arena
    new_index = loc_to_index(loc, L) # new location index
    return new_index

def compute_adjacency(walls):
    num_actions = len(action_deltas)
    N = walls.shape[0]
    L = int(np.sqrt(N))
    adjacency = torch.zeros(N, N) # start with no transitions
    neighbors = torch.zeros(N, num_actions, dtype = int) # array to store neighbors
    for i in range(N): # for each location
        adjacency[i,i] = 1
        neighbors[i, -1] = i # standing still transitions to self
        for a in range(walls.shape[-1]): # for each action
            if walls[i, a] == 0: # if no wall
                j = ind_a_ind(i, a, L) # neighboring state
                adjacency[i, j] = 1 # allowed transition
                adjacency[j, i] = 1 # allowed transition
                neighbors[i, a] = j
            else:
                neighbors[i, a] = i
    return adjacency, neighbors

