
"""Code for plotting all schematics in Figure 2"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
import pysta
from pysta.maze_utils import loc_to_index
from pysta.plot_utils import Arrow3D
from pysta.plot_utils import plot_perspective_attractor
from scipy.sparse.csgraph import dijkstra
import matplotlib.patches as patches
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from pysta import basedir

ext = ".pdf"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8


#%% Set some parameters

styleA = "Simple, tail_width=0.5, head_width=4.5, head_length=5"
styleB = "|-|, widthB=3.5, angleA=0, widthA=0, angleB=0"

cmap_connections = "coolwarm"
cmap_activity = "YlOrRd"

#%% Plot ring attractor schematic
thetas = np.linspace(0.5*np.pi, 2.5*np.pi, 17)[:-1]
cell_locs = np.array([np.cos(thetas), np.sin(thetas)]).T

col_ref = (0.35, 0.65, 0.2)

col = [0.85, 0.85, 0.9]
col2 = [0.9, 0.8, 0.8]
surface_col = [0.6, 0.6, 0.6]

arrows = []
loc0, theta0 = cell_locs[0], thetas[0]
for iloc, loc in enumerate(cell_locs[1:]):
    dtheta = thetas[iloc+1] - theta0
    dtheta = dtheta if dtheta <= np.pi else dtheta - 2*np.pi
    ex = np.abs(dtheta) < np.pi/2
    style = styleA if ex else styleB
    kw = dict(arrowstyle=style, color=plt.get_cmap(cmap_connections)(1-np.abs(dtheta)/np.pi))
    constyle = "arc3,rad="+str(-np.sign(dtheta)*0.7*(1-np.abs(dtheta/np.pi)))
    a = patches.FancyArrowPatch(0.91*loc0, 0.91*loc,
                             connectionstyle=constyle, **kw, lw = 0.9 if ex else 1.65)

    arrows.append(a)

plt.figure(figsize = (3.55/2.54, 3.55/2.54))
plt.scatter(cell_locs[:, 0], cell_locs[:, 1], marker = "o",
            color = col, s = 70, edgecolors = [0.6, 0.6, 0.6])
plt.scatter(cell_locs[:1, 0], cell_locs[:1, 1], marker = "o",
            color = col_ref, s = 70, edgecolors = [0.6, 0.6, 0.6])
for a in arrows:
    plt.gca().add_patch(a)
plt.gca().axis("off")
plt.xlim(-1.15, 1.15)
plt.ylim(-1.15, 1.15)
plt.tight_layout = True
plt.savefig(f"{basedir}/figures/schematics/fig2_hd_connections{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
xs = np.arange(len(thetas))
refs = [3, len(thetas)-4]
acts = [[np.cos(theta - thetas[ref]) for theta in thetas]for ref in refs]
fig, axs = plt.subplots(2,1, figsize = (3.35/2.54, 3.35/2.54))
for iax, ax in enumerate(axs):
    for ix, act in enumerate(acts[iax]):
        ax.bar(xs[ix], [act], color = plt.get_cmap(cmap_activity)(act))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("firing rate", labelpad = 2.5)
    ax.set_ylim(0, 1.3)
axs[1].set_xlabel(r"preferred $\theta$", labelpad = 3.5)
plt.tight_layout = True
plt.savefig(f"{basedir}/figures/schematics/fig2_hd_activity{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()



#%% Plot grid attractor schematic

plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
xs = np.linspace(0, 1, 11)
cell_locs = np.array([locs.flatten() for locs in np.meshgrid(xs, xs)]).T
cent = np.argmin(((cell_locs - np.ones(2)[None, :]*0.5)**2).sum(-1))
sgrid = 24
plt.figure(figsize = (3.35/2.54, 3.35/2.54))
plt.scatter(cell_locs[:, 0], cell_locs[:, 1], marker = "o",
            color = col, s = sgrid, edgecolors = [0.6, 0.6, 0.6], lw = 0.5)
plt.scatter(cell_locs[cent:cent+1, 0], cell_locs[cent:cent+1, 1], marker = "o",
            color = col_ref, s = sgrid, edgecolors = [0.6, 0.6, 0.6], lw = 0.5)

xs_im = np.linspace(-0.05, 1.05, 61)
imshow_grid = np.meshgrid(xs_im, xs_im)
imshow_dists = (imshow_grid[0] - cell_locs[cent, 0])**2 + (imshow_grid[1] - cell_locs[cent, 1])**2
imshow_vals = np.exp(-(16*imshow_dists)**2) - 0.7*np.exp(-(10*imshow_dists)**2)
vmax = np.amax(np.abs(imshow_vals))
plt.imshow(imshow_vals, extent = (-0.05, 1.05, -0.05, 1.05), cmap = cmap_connections, vmin = -vmax, vmax = vmax)
plt.xticks([])
plt.yticks([])
plt.xlabel("preferred x loc", labelpad = 3.5)
plt.ylabel("preferred y loc", labelpad = 2.5)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.tight_layout = True
plt.savefig(f"{basedir}/figures/schematics/grid_connections{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

x,y = np.abs(cell_locs - cell_locs[cent:cent+1, :]).T*16

act = np.maximum(np.cos(x) + np.cos(0.5*(x + np.sqrt(3)*y)) + np.cos(0.5*(-x+np.sqrt(3)*y)), 0)

act = act**1.1 / np.amax(act**1.1)

plt.figure(figsize = (3.35/2.54, 3.35/2.54))
plt.scatter(cell_locs[:, 0], cell_locs[:, 1], marker = "o",
            color = plt.get_cmap(cmap_activity)(act), s = sgrid, edgecolors = [0.6, 0.6, 0.6], lw = 0.5)
plt.xticks([])
plt.yticks([])
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xlabel("preferred x loc", labelpad = 3.5)
plt.ylabel("preferred y loc", labelpad = 2.5)
plt.tight_layout = True
plt.savefig(f"{basedir}/figures/schematics/grid_activity{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%% now plot PFC attractor fixed points

from pysta.utils import schematic_walls as walls
num_locs = walls.shape[0]
L = int(np.sqrt(num_locs))
num_mod = 4


all_dists = torch.tensor(dijkstra(pysta.maze_utils.compute_adjacency(walls)[0], directed=False, unweighted = True))

all_locs = [[(0,1), (1,1), (1,0), (2,0)],
            [(2,1), (1,1), (0,1), (0,2)]] # trajectories

for ipath, locs in enumerate(all_locs):
    path_inds = [loc_to_index(loc, L) for loc in locs]
    
    vmap = torch.zeros(num_mod, num_locs)
    for z in range(4):
        vmap[z, :] = np.exp(-1.7*all_dists[path_inds[z], :]**2)

    pysta.reload()
    ax = pysta.plot_utils.plot_perspective_attractor(walls, vmap, filename = None, plot_proj = False, cmap = "YlOrRd", vmin = -0.15, vmax = 1.05, figsize = (3.5,2.2), aspect = (1,1,2.2), view_init = (-22,-10,-90))
    plt.savefig(f"{basedir}/figures/schematics/pfc_activity{ipath}{ext}", bbox_inches = mpl.transforms.Bbox([[0.90,0.55], [2.7,1.65]]), transparent = True)
    plt.show()
    plt.close()


#%% plot connectivity with arrows etc.


# on path
stim = (2, 1, 0)
excite = [(1,1,1), (1,2,0), (1,1,0), (3,1,1), (3,2,0), (3,1,0)]
inhibit = [(2, x // 3, x % 3) for x in range(9) if not (x%3 == 0 and x // 3 == 1)]

act_cols = [[np.ones(3)*0.92 for _ in range(9)] for _ in range(4)]

for a in excite:
    act_cols[a[0]][loc_to_index([a[1], a[2]], L = 3)] = plt.get_cmap("coolwarm")(0.95)
for i in inhibit:
   act_cols[i[0]][loc_to_index([i[1], i[2]], L = 3)] = plt.get_cmap("coolwarm")(0.05)
act_cols[stim[0]][loc_to_index([stim[1], stim[2]], L = 3)] = (0.35, 0.65, 0.2)


vmap = np.zeros((4, 9))
ax = plot_perspective_attractor(walls, vmap, state_actions = False, act_cols = act_cols, cmap = "coolwarm", plot_proj = False, figsize = (3.5,2.2), aspect = (1,1,2.2), view_init = (-22,-10,-90), plot_subs = True, show = False)

styleA = "Simple, tail_width=2.2, head_width=6, head_length=8"
styleB = "Simple, tail_width=2.2, head_width=1.5, head_length=0.1"
exc_col = plt.get_cmap("coolwarm")(0.9)
propA = dict(mutation_scale=1, arrowstyle=styleA, lw=0.0, color=exc_col, shrinkA=0, shrinkB=0, alpha = 0.8)
propB = dict(mutation_scale=1, arrowstyle=styleB, lw=0.0, color=exc_col, shrinkA=0, shrinkB=0, alpha = 0.8)

# new
a = Arrow3D([stim[1], 0.80], [2-stim[2], 2-0.59], [stim[0], 1.43], **propB, zorder = 100)
ax.add_artist(a)
a = Arrow3D([stim[1], 0.2], [2-stim[2], 2+0.51], [stim[0], 1.5], **propA, zorder = 100)
ax.add_artist(a)
a = Arrow3D([stim[1], 1.0], [2-stim[2]-0.18, 2-0.99], [stim[0]+0.08, 3.0], **propA, zorder = 100)
ax.add_artist(a)
a = Arrow3D([stim[1]+0.3, 2.1], [2-stim[2]+0.1, 2+0.05], [stim[0]+0.3, 2.98], **propA, zorder = 100)
ax.add_artist(a)

a = Arrow3D([stim[1], 0.4], [2-stim[2], 2+0.18], [stim[0], 1.51], **propB, zorder = 100)
ax.add_artist(a)
a = Arrow3D([stim[1]+0.3, 1.95], [2-stim[2]+0.04, 2-0.26], [stim[0]+0.3, 2.70], **propA, zorder = 100)
ax.add_artist(a)

plt.tight_layout = True
plt.savefig(f"{basedir}/figures/schematics/pfc_connectivity_arrows{ext}", bbox_inches = mpl.transforms.Bbox([[0.85,0.5], [2.75,1.7]]), transparent = True)

plt.show()
plt.close()


#%% plot sequence of three PFC representations

locs = [(0,1), (1,1), (1,0), (2,0)] # trajectory
path_inds = [loc_to_index(loc, L) for loc in locs]

vmap_init = np.random.uniform(0, 0.2, size = (num_mod, num_locs))
vmap_plan = torch.zeros(num_mod, num_locs)
for z in range(4):
    vmap_plan[z, :] = np.exp(-1.7*all_dists[path_inds[z], :]**2)
vmap_move = torch.zeros(num_mod, num_locs)
for z in range(3):
    vmap_move[z, :] = np.exp(-1.7*all_dists[path_inds[z+1], :]**2)
vmap_move[-1, :] = vmap_move[-2, :]

    
for istep, vmap in enumerate([vmap_init, vmap_plan, vmap_move]):

    ax = pysta.plot_utils.plot_perspective_attractor(walls, vmap, filename = None, plot_proj = False, cmap = "YlOrRd", vmin = -0.15, vmax = 1.05, figsize = (3.5,2.2), aspect = (1,1,2.2), view_init = (-22,-10,-90))
    plt.savefig(f"{basedir}/figures/schematics/pfc_trajectory{istep}{ext}", bbox_inches = mpl.transforms.Bbox([[0.90,0.55], [2.7,1.65]]), transparent = True)
    plt.show()
    plt.close()


