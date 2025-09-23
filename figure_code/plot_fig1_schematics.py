"""Code for plotting all schematics in Figure 1"""

#%%

import numpy as np
import torch
import pysta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import os
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

#%% load the example maze parameters

from pysta.utils import schematic_walls as walls
adjacency, neighbors = pysta.maze_utils.compute_adjacency(walls)
dists = pysta.maze_utils.compute_shortest_dists(adjacency)
N, L = walls.shape[0], int(np.sqrt(walls.shape[0]))

#%% plot maze and fully connected scaffolds

locs = [(0,1), (1,1), (1,0), (2,0)]
loc_inds = np.array([pysta.maze_utils.loc_to_index(loc, L = L) for loc in locs])
dists_to_locs = dists[loc_inds, :]
acts = np.exp(-1.7*dists_to_locs**2)

size = (1.16, 1.16)
plt.figure(figsize = size)
ax = plt.gca()
pysta.plot_utils.plot_maze_scaffold(adjacency, ax = ax, s = 280, lw = 5.5)
ax.axis("off")
plt.savefig(f"{basedir}/figures/schematics/scaffold{ext}", bbox_inches = "tight", transparent = True)
plt.show()

plt.figure(figsize = size)
ax = plt.gca()
pysta.plot_utils.plot_maze_scaffold(pysta.maze_utils.compute_adjacency(walls*0)[0], ax = ax, s = 280, lw = 5.5)
ax.axis("off")
plt.savefig(f"{basedir}/figures/schematics/scaffold_connected{ext}", bbox_inches = "tight", transparent = True)
plt.show()


# %% Plot example PFC representation

sequence_colors = [plt.get_cmap("viridis")(iind / 5 + 0.35) for iind in range(4)][::-1]

locs = [(0,1), (1,1), (1,0), (2,0)]
loc_inds = np.array([pysta.maze_utils.loc_to_index(loc, L = L) for loc in locs])
dists_to_locs = dists[loc_inds, :]
acts = np.exp(-1.7*dists_to_locs**2)

pysta.reload()
edgecolors = [[1,1,1,0] for _ in range(acts.size)]

sta_sequence_inds = [1, 13, 21, 33]
for iind, ind in enumerate(sta_sequence_inds):
    edgecolors[ind] = sequence_colors[iind]

ax = pysta.plot_utils.plot_perspective_attractor(walls, acts, cmap = "YlOrRd", plot_proj = False, figsize = (5.4, 2.2), aspect = (1,1,2.5), view_init = (-25, -10, -90), plot_subs = True, show = False, edgecolors = edgecolors)
plt.savefig(f"{basedir}/figures/schematics/sta_cells{ext}", bbox_inches = mpl.transforms.Bbox([[1.92,0.6], [3.62,1.6]]), transparent = True)
plt.show()
plt.close()

#%% plot colorbar

plt.figure(figsize = (0.4,3))
plt.imshow(np.linspace(1,0, 101)[:, None], vmin = 0, vmax = 1.1, aspect = "auto", cmap = "YlOrRd")
plt.xticks([])
plt.yticks([])
plt.savefig(f"{basedir}/figures/schematics/fr_cbar{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


#%% now plot spikes over time

ys = np.arange(4)
xs = np.zeros(len(ys))+0.5

plt.figure(figsize = (0.4,0.9))
for i in range(len(ys)):
    plt.plot([xs[i], xs[i]], [-ys[i]-0.33, -ys[i]+0.33], color = sequence_colors[i], lw = 3)

plt.xlim(0, 1)
plt.ylim(-3.8, 0.8)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.xlabel("time", labelpad = 3.5)
plt.ylabel("neuron", labelpad = 2.5)
plt.gca().spines[['left', 'bottom']].set_linewidth(1.3)
plt.savefig(f"{basedir}/figures/schematics/raster{ext}", bbox_inches = "tight", transparent = True)
plt.show()

#%% plot visual cortex inference example

from pysta.V1_panel_utils import V1_inference_data, arc_patch

xs = np.mean(V1_inference_data[..., 0], axis = (0,1)) # align
ys = V1_inference_data[:, 0, :, 1] # mean
# errors
err = np.array([np.abs(V1_inference_data[:,i, :, 1]) - ys for i in range(2)]).mean(0)

for ictrl in range(3):
    plt.figure(figsize = (1.90,0.52))
    for i in range(ictrl+1):
        plt.plot(xs, ys[i])
        plt.fill_between(xs[i], ys[i]-err[i], ys[i]+err[i], alpha = 0.2, linewidth = 0)
    plt.xlim(-1, +1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("receptive field alignment", labelpad = 3.5)
    plt.ylabel("V1 activity", labelpad = 2.5)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.gca().spines[['left', 'bottom']].set_linewidth(1.3)
    plt.savefig(f"{basedir}/figures/schematics/V1_response{ictrl}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

#%% now plot the stimuli

theta1 = 0
theta2 = 6/4*np.pi
radius = 0.6

centers = [(1,1), (1,-1), (-1,-1), (-1,1)]

for ictrl in range(3):
    col = plt.get_cmap("tab10")(ictrl)

    plt.figure(figsize = (0.72, 0.72))
    ax = plt.gca()
    for icenter, center in enumerate(centers):

        if ictrl == 1:
            rotate = -(icenter+3)*np.pi/2
        else:
            rotate = -(icenter+1)*np.pi/2

        arc_patch(center, radius, theta1, theta2, rotate = rotate, ax = ax, fc = col, ec = col, lw = 0.1)

        if ictrl == 2:
            circle = plt.Circle(center, radius, fill=None, color = col, lw=2)
            ax.add_patch(circle)
    plt.xlim(-1.6, 1.6)
    plt.ylim(-1.6, 1.6)
    ax.axis("off")
    plt.savefig(f"{basedir}/figures/schematics/stim{ictrl}{ext}", bbox_inches = "tight", transparent = True)
    plt.show()
    plt.close()

# %% now plot ring attractor

styleA = "Simple, tail_width=0.5, head_width=3.5, head_length=4"
styleB = "|-|, widthB=2.7, angleA=0, widthA=0, angleB=0"

thetas = np.linspace(0.5*np.pi, 2.5*np.pi, 17)[:-1]
cell_locs = np.array([np.cos(thetas), np.sin(thetas)]).T

arrows = []
loc0, theta0 = cell_locs[0], thetas[0]
for iloc, loc in enumerate(cell_locs[1:]):
    dtheta = thetas[iloc+1] - theta0
    dtheta = dtheta if dtheta <= np.pi else dtheta - 2*np.pi
    ex = np.abs(dtheta) < np.pi/2
    style = styleA if ex else styleB
    kw = dict(arrowstyle=style, color=plt.get_cmap("coolwarm")(1-np.abs(dtheta)/np.pi))
    constyle = "arc3,rad="+str(-np.sign(dtheta)*0.7*(1-np.abs(dtheta/np.pi)))
    a = patches.FancyArrowPatch(0.91*loc0, 0.91*loc,
                             connectionstyle=constyle, **kw, lw = 0.8 if ex else 1.14)

    arrows.append(a)

plt.figure(figsize = (2.4/2.54, 2.4/2.54))
plt.scatter(cell_locs[:, 0], cell_locs[:, 1], marker = "o",
            color = np.ones(3)*0.7, s = 32, edgecolors = np.ones(3)*0.5, lw = 0.8)
for a in arrows:
    plt.gca().add_patch(a)
plt.gca().axis("off")
plt.xlim(-1.15, 1.15)
plt.ylim(-1.15, 1.15)
plt.tight_layout = True
plt.savefig(f"{basedir}/figures/schematics/fig1_hd_connections{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

thetas_act = np.linspace(0.5*np.pi, 2.5*np.pi, 30)[:-1]
xs = np.arange(len(thetas_act))
ring_acts = [np.cos(theta - thetas_act[9]) for theta in thetas_act]
fig = plt.figure(figsize = (2.7/2.54, 1.6/2.54))
ax = plt.gca()
for ix, act in enumerate(ring_acts):
    ax.bar(xs[ix], [act], color = plt.get_cmap("YlOrRd")(act))

plt.xticks([])
plt.yticks([])
plt.ylabel("firing rate", labelpad = 2.5)
plt.xlabel("preferred angle", labelpad = 3.5)
plt.ylim(0, 1.1)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.gca().spines[['left', 'bottom']].set_linewidth(1.5)

plt.tight_layout = True
plt.savefig(f"{basedir}/figures/schematics/fig1_hd_activity{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


# %%
