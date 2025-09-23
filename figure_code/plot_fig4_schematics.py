"""Code for plotting all schematics in Figure 4"""

#%% load libraries

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import matplotlib as mpl
import torch
from scipy.sparse.csgraph import dijkstra
from pysta import basedir
from scipy.ndimage import gaussian_filter1d
from matplotlib import patches
ext = ".pdf"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

    
#%% plot scaffolds with trajectories on top
ex_seed = 31
example_model = f"MazeEnv_L4_max6/landscape_changing-rew_dynamic-rew_constant-maze/allo_planrew_plan5-6-7/VanillaRNN/iter10_tau5.0_opt/N800_linout/model{ex_seed}"
rnn = pysta.utils.load_model(example_model)[0]
walls = rnn.env.walls[0].numpy()

num_locs = walls.shape[0]
L = int(np.sqrt(num_locs))
adjacency = pysta.maze_utils.compute_adjacency(walls)[0].numpy()
all_dists = dijkstra(pysta.maze_utils.compute_adjacency(walls)[0], directed=False, unweighted = True)

all_locs = [
    [(1,2), (2,2), (3,2), (3,1), (3,0)],
    [(2,2), (3,2), (3,1), (3,0)],
    [(2,0), (3,0), (3,1), (3,2)]
]


istarts = [0, 1, 0]
for ilocs, locs in enumerate(all_locs):
    locs = np.array(locs).T.astype(float)

    plt.figure(figsize = (1.43,1.43))
    ax = plt.gca()
    pysta.plot_utils.plot_maze_scaffold(adjacency, ax = ax, s = 250, lw = 5)

    smooth_locs = gaussian_filter1d(np.repeat(locs, 6, axis = -1), 2, axis = -1, mode = "nearest").T
    chunks = np.array_split(np.arange(len(smooth_locs)), locs.shape[-1])
    cols = [plt.get_cmap("viridis")(0.7*(i)/5 + 0.40) for i in range(5)][::-1][istarts[ilocs]:]
    for ichunk, chunk in enumerate(chunks):
        if ichunk < len(chunks)-1:
            chunk = np.concatenate([chunk, chunks[ichunk+1][:1]])
        ax.plot(smooth_locs[chunk, 0], smooth_locs[chunk, 1], color = cols[ichunk], lw = 6)

    ax.axis("off")
    plt.savefig(f"{basedir}/figures/rnn_decoding/maze_scaffold{ilocs}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()
    
    
# %% plot some example representations

num_mod = 4

for ipath, locs in enumerate(all_locs):
    num_mod = min(num_mod, len(locs))
    locs = locs[:num_mod]
    
    path_inds = [pysta.maze_utils.loc_to_index(loc, L) for loc in locs if len(loc) == 2]
    
    vmap = np.zeros((num_mod, num_locs))
    for z in range(len(locs)):
        if z < len(path_inds):
            vmap[z, :] = np.exp(-1.7*all_dists[path_inds[z], :]**2)
        else:
            vmap[z, :] = np.exp(-1.7*np.amin(all_dists[adjacency[path_inds[z-1], :].astype(bool), :], 0)**2)/adjacency[path_inds[z-1], :].sum()

    edgecolors = [[1,1,1,0] for _ in range(vmap.size)]
            
    pysta.reload()
    ax = pysta.plot_utils.plot_perspective_attractor(walls, vmap, edgecolors = edgecolors, filename = None, plot_proj = False, cmap = "YlOrRd", vmin = -0.15, vmax = 1.05, figsize = (3.5,2.2), aspect = (1,1,2.2), view_init = (-22,-10,-90))
    plt.savefig(f"{basedir}/figures/rnn_decoding/pfc_activity{ipath}{ext}", bbox_inches = mpl.transforms.Bbox([[0.90,0.55], [2.7,1.65]]), transparent = True)
    plt.show()
    plt.close()
    

 #%% and plot an example 'space-by-time' reward function

np.random.seed(1)
torch.manual_seed(1)

env = pysta.envs.MazeEnv(side_length = rnn.env.side_length, max_steps = 3, changing_trial_rew = True, batch_size = 1)
env.walls[0, ...] = torch.tensor(walls[...])

path = [6, 10, 14,13]
optimal = False
# make sure to sample a value function that shares the same optimal path
while not optimal:
    env.reset()
    vs = env.vs[0].detach().numpy()
    optimal = True
    for i in range(len(path)-1):
        adjacent_locs = np.where(adjacency[path[i]])[0]
        if adjacent_locs[np.argmax(vs[i+1, adjacent_locs])] != path[i+1]:
            optimal = False

vs = env.vs.detach().numpy()[0]
vs = ((vs - vs.mean(-1)[:, None]) / vs.std(-1)[:, None]) # normalize within each subspace

ax = pysta.plot_utils.plot_perspective_attractor(walls, vs, vmin = -1.5, vmax = 2, filename = None, plot_proj = False, cmap = "YlOrRd", figsize = (3.5,2.2), aspect = (1,1,2.2), view_init = (-22,-10,-90))
plt.savefig(f"{basedir}/figures/rnn_decoding/value_function{ext}", bbox_inches = mpl.transforms.Bbox([[0.90,0.55], [2.7,1.65]]), transparent = True)
plt.show()
plt.close()


#%% plot an example RNN

styleA = "Simple, tail_width=0.5, head_width=3.5, head_length=4"
styleB = "|-|, widthB=2.7, angleA=0, widthA=0, angleB=0"

cell_locs = np.array([
    [3,8],
    [5.5,2],
    [8,4],
    [2,4],
    [3.2,2],
    [7.8,6],
    [6,8],
    [4.7,5.2]
])
cell_locs = cell_locs[np.argsort(cell_locs[:, 0])]

minv, maxv = np.amin(cell_locs, axis = 0), np.amax(cell_locs, axis = 0)

cell_locs = (cell_locs - minv[None, :])/(maxv-minv)[None, :]*2-1 # shift to origin

plt.figure(figsize = (1.5,1.5))
plt.scatter(cell_locs[:,0], cell_locs[:,1], s = 160, color = np.ones(3)*0.7, edgecolor = np.ones(3)*0.5)


cons = [
    (0, 3, -0.2, "+"),
    (2,6, -0.1, "-"),
    (1,4, 0.2, "+"),
    (7,1,0.1,"-"),
    (5,4,-0.1,"+"),
    (1,0,0.2,"-"),
    (4,7,0.2,"+"),
    (6,5,0.2,"+"),
    (3,5,-0.2,"-")
]

for con in cons:
    loc0, loc1, rad = np.array(cell_locs[con[0]]), np.array(cell_locs[con[1]]), con[2]
    dloc = loc1 - loc0
    dloc = dloc / np.sqrt(np.sum(dloc**2)) # normalize
    ex = True if con[3] == "+" else False
    style = styleA if ex else styleB
    
    kw = dict(arrowstyle=style, color=plt.get_cmap("coolwarm")(0.8 if ex else 0.2))
    
    constyle = "arc3,rad="+str(rad)
    a = patches.FancyArrowPatch(loc0+0.18*dloc, loc1-0.15*dloc,
                             connectionstyle=constyle, **kw, lw = 1.3 if ex else 1.7)
    plt.gca().add_patch(a)


plt.xlim(-1.2, 1.2)
plt.ylim(-1.2,1.2)
plt.gca().axis("off")
plt.savefig(f"{basedir}/figures/rnn_decoding/rnn_schematic{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()
    
    
# %%
