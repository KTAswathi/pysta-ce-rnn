"""Code for plotting all schematics in Figure 5"""

#%% load libraries
import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import torch
import os
import matplotlib as mpl
from pysta import basedir
from scipy.spatial.transform import Rotation
ext = ".pdf"
basefigdir = f"{basedir}/figures/rnn_connectivity/"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8


# %% plot example projections

np.random.seed(8)
ts = np.linspace(0,1, 101)

K = np.exp(-(ts[:, None] - ts[None, :])**2/(0.15**2))
L = np.linalg.cholesky(K + np.eye(len(K))*1e-10)

X = L @ np.random.normal(0, 1, (len(ts), 2))
X[:, 1] *= -1
Z1 = np.concatenate([ts[:, None], X], axis = -1)

figsize = (1.15, 1.15)
ax = plt.figure(figsize = figsize).add_subplot(projection='3d')
ax.plot3D(Z1[:, 0], Z1[:, 1], Z1[:, 2], lw = 2.5)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_zlabel("y")
zmin, zmax = np.min(Z1[:, 2]), np.max(Z1[:, 2])
delta = zmax - zmin
ax.set_zlim(zmin - 0.1*delta, zmax + 0.1*delta)
ax.axis("off")
ax.view_init(20, -75, 0)
plt.savefig(f"{basedir}/figures/rnn_connectivity/traj3D_aligned{ext}", transparent = True, bbox_inches = "tight")
ax.axis("on")
plt.show()
plt.close()

ax = plt.figure(figsize = figsize).add_subplot(projection='3d')
ax.plot3D(Z1[:, 0], Z1[:, 1], 0*Z1[:, 2]+zmin-0.099*delta, lw = 0.6)
ax.set_zlim(zmin - 0.1*delta, zmax + 0.1*delta)
ax.axis("off")
ax.view_init(20, -75, 0)
plt.savefig(f"{basedir}/figures/rnn_connectivity/traj2D_aligned{ext}", transparent = True, bbox_inches = "tight")
ax.axis("on")
plt.show()
plt.close()

q = np.random.normal(0, 1, 4)
q = q / np.sqrt(np.sum(q**2))
rot = Rotation.from_quat(q)
Z2 = (rot.as_matrix() @ Z1.T).T

ax = plt.figure(figsize = figsize).add_subplot(projection='3d')
ax.plot3D(Z2[:, 0], Z2[:, 1], Z2[:, 2], lw = 2.5)
zmin, zmax = np.min(Z2[:, 2]), np.max(Z2[:, 2])
delta = zmax - zmin
ax.set_zlim(zmin - 0.1*delta, zmax + 0.1*delta)
ax.axis("off")
ax.view_init(20, -75, 0)
plt.savefig(f"{basedir}/figures/rnn_connectivity/traj3D_nonaligned{ext}", transparent = True, bbox_inches = "tight")
ax.axis("on")
plt.show()
plt.close()

ax = plt.figure(figsize = figsize).add_subplot(projection='3d')
ax.plot3D(Z2[:, 0], Z2[:, 1], 0*Z2[:, 2]+zmin-0.099*delta, lw = 0.6)
ax.set_zlim(zmin - 0.1*delta, zmax + 0.1*delta)
ax.axis("off")
ax.view_init(20, -75, 0)
plt.savefig(f"{basedir}/figures/rnn_connectivity/traj2D_nonaligned{ext}", transparent = True, bbox_inches = "tight")
ax.axis("on")
plt.show()
plt.close()

# %% load data
ex_seed = 31
example_model = f"sta_MazeEnv_L4_max6_landscape_changing-rew_dynamic-rew_constant-maze_allo_planrew_plan5-6-7_VanillaRNN_iter10_tau5.0_opt_N800_linout_model{ex_seed}"
connectivity_data = pickle.load(open(f"{basedir}/data/rnn_analyses/{example_model}_connectivity_data.pickle", "rb"))

env = connectivity_data["rnn"].env
walls = env.walls[0]
rs1 = env.rews[0][1]
adj = env.adjacency[0].detach().numpy()
stim_col = (0.35, 0.65, 0.2)

vmap = np.zeros((3, env.num_locs))
vmap += np.random.uniform(0.15,0.35, vmap.shape)

#%% plot a colorbar

plt.figure(figsize = (0.6,4))
plt.imshow(np.linspace(+1, -1, 101)[:, None], vmin = -1, vmax = +1, aspect = "auto", cmap = "coolwarm")
plt.xticks([])
plt.yticks([])
plt.savefig(f"{basedir}/figures/rnn_connectivity/conn_cbar{ext}", bbox_inches = "tight", transparent = True)
plt.show()

#%% example 'connection'

ind0, ind1 = 6, 7
act_cols = [[np.ones(3)*0.92 for _ in range(env.num_locs)] for _ in range(3)]
act_cols[0][ind0] = stim_col
act_cols[1][ind1] = (0.95, 0.1, 0.95)

bbox = mpl.transforms.Bbox([[2.1,1.0], [5.15,2.95]])
pysta.plot_utils.plot_perspective_attractor(walls, vmap[:2],  act_cols = act_cols, vmin = 0, vmax = 1, cmap = "coolwarm",
                                            lw = 4, plot_proj = False, figsize = (7,4), aspect = (1,1,2.0), view_init = (-35,-10,-90),
                                            bbox_inches = bbox, transparent = True, filename = f"{basefigdir}schematic_coordinates{ext}", show = True)


#%% schematic weights

adj_inds = np.where(adj[ind0, :] > 0)[0]
for aind in adj_inds:
    act_cols[1][aind] = plt.get_cmap("coolwarm")(0.95)

bbox = mpl.transforms.Bbox([[2.1,1.0], [5.15,2.95]])
pysta.plot_utils.plot_perspective_attractor(walls, vmap[:2],  act_cols = act_cols, vmin = 0, vmax = 1, cmap = "coolwarm",
                                            lw = 4, plot_proj = False, figsize = (7,4), aspect = (1,1,2.0), view_init = (-35,-10,-90),
                                            bbox_inches = bbox, transparent = True, filename = f"{basefigdir}schematic_weights{ext}", show = True)
                


#%% scaffold and 'empty' STA

plt.figure(figsize = (2.4,2.4))
ax = plt.gca()
pysta.plot_utils.plot_maze_scaffold(adj, ax = ax)
ax.axis("off")
plt.scatter([1,], [2,], marker = ".", color = stim_col, s = 1600, zorder = 200)
plt.savefig(f"{basefigdir}schematic_maze_scaffold{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

pysta.plot_utils.plot_perspective_attractor(walls, vmap, vmin = 0, vmax = 1,
                                            lw = 4, plot_proj = False, figsize = (7,4), aspect = (1,1,2.0), view_init = (-30,-10,-90),
                                            bbox_inches = bbox, transparent = True, filename = f"{basefigdir}schematic_activity{ext}", show = True)
                


#%% future rewards

vmin, vmax, cmap = -1.2,2.2,"Greens"
plt.figure(figsize = (1.0, 1.0))
ax = plt.gca()
pysta.plot_utils.plot_flat_frame(walls, ax = ax, filename = None, vmap = rs1, cmap = cmap, vmin = vmin, vmax = vmax, show = False, lw = 3.0)
plt.scatter([2,], [1,], marker = ".", color = stim_col, s = 380, zorder = 200)
plt.savefig(f"{basefigdir}schematic_rewards{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()



# %%
