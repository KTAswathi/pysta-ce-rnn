"""Code for plotting all schematics in Figure 6"""

#%% load libraries
import pysta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pysta import basedir
from pysta.maze_utils import loc_to_index
from pysta.plot_utils import plot_perspective_attractor
ext = ".pdf"
basefigdir = f"{basedir}/figures/changing_maze_rnn/"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%% set some parameters
cols = {"stim": (0.35, 0.65, 0.2),
          "strong_ex": plt.get_cmap("coolwarm")(0.95), 
          "weak_ex": plt.get_cmap("coolwarm")(0.55),
          "strong_inh": plt.get_cmap("coolwarm")(0.05),
          "weak_inh": plt.get_cmap("coolwarm")(0.35),
          "neutral": plt.get_cmap("coolwarm")(0.44),}

walls = np.zeros((9, 4))
walls[0,:], walls[3,:] = 1.0, 1.0
Nloc = walls.shape[0]
L = int(np.sqrt(Nloc))
Nmod = 3

#%% plot overall schematic

stim = (1, 1, 0)
excite = [(2,1,1), (2,2,0)]
not_excite = [(2,0,0)]

act_cols = [[[np.ones(3)*0.92 for _ in range(4)] for _ in range(Nloc)] for _ in range(Nmod)] # for each transition for each location for each module

for a in excite:
    for itrans in range(4):
        act_cols[a[0]][loc_to_index([a[1], a[2]], L = 3)][itrans] = cols["strong_ex"]
for a in not_excite:
    for itrans in range(4):
        act_cols[a[0]][loc_to_index([a[1], a[2]], L = 3)][itrans] = cols["weak_ex"]
        
act_cols[0][4][1] = cols["strong_ex"]
act_cols[0][6][0] = cols["strong_ex"]
act_cols[0][0][2] = cols["weak_ex"]

for itrans in [2,3]:
    act_cols[stim[0]][loc_to_index([stim[1], stim[2]], L = 3)][itrans] = cols["stim"]
for itrans in [0]:
    act_cols[stim[0]][loc_to_index([stim[1], stim[2]], L = 3)][itrans] = cols["strong_inh"]

vmap = np.zeros((Nmod, Nloc, 4))
ax = plot_perspective_attractor(walls, vmap, state_actions = True, act_cols = act_cols, cmap = "coolwarm", plot_proj = False, plot_subs = True, show = False,
                                figsize = (5.5, 3.1), aspect = (1,1,3.9), view_init = (-50, -10, -90))

for mod in range(Nmod):
    ax.plot([0.5,0.5], [1.8,2.2], [mod, mod], color = "k", lw = 2)


plt.tight_layout = True
plt.savefig(f"{basefigdir}schematic{ext}", bbox_inches = mpl.transforms.Bbox([[1.75,1.05], [3.8,2.05]]), transparent = True)
plt.show()
plt.close()

# %% plot schematic for connection strength analysis

# on path
walls_all = np.zeros(walls.shape)

act_cols = [[[np.ones(3)*0.92 for _ in range(4)] for _ in range(Nloc)] for _ in range(Nmod)] # for each transition for each location for each module
        
act_cols[0][7][3] = cols["stim"]
for i in range(4):
    act_cols[1][8][i] = cols["strong_ex"]

    for loc in [4,6,7]:
        act_cols[1][loc][i] = cols["neutral"]


vmap = np.zeros((2, Nloc, 4))
ax = plot_perspective_attractor(walls_all, vmap, state_actions = True, act_cols = act_cols, cmap = "coolwarm", plot_proj = False, plot_subs = True, show = False,
                                figsize = (5.5, 3.1), aspect = (1,1,2.3), view_init = (-50, -10, -90))

plt.tight_layout = True
plt.savefig(f"{basefigdir}connections{ext}", bbox_inches = mpl.transforms.Bbox([[1.7,0.80], [3.85,2.3]]), transparent = True)
plt.show()
plt.close()


#%% plot schematic for the 'transition' input analysis

# on path
walls_all = np.zeros(walls.shape)

act_cols = [[[np.ones(3)*0.92 for _ in range(4)] for _ in range(Nloc)] for _ in range(Nmod)] # for each transition for each location for each module

Nmod = 2
for mod in range(Nmod):
    for i in range(4):
        for loc in [4,7]:
            act_cols[mod][loc][i] = cols["neutral"]
    act_cols[mod][4][2] = cols["strong_inh"]
    act_cols[mod][7][0] = cols["strong_inh"]


vmap = np.zeros((Nmod, Nloc, 4))
ax = plot_perspective_attractor(walls_all, vmap, state_actions = True, act_cols = act_cols, cmap = "coolwarm", plot_proj = False, plot_subs = True, show = False,
                                figsize = (5.5, 3.1), aspect = (1,1,2.3), view_init = (-50, -10, -90))

for mod in range(Nmod):
    ax.plot([1.5,1.5], [0.7,1.3], [mod, mod], color = "k", lw = 4)

plt.tight_layout = True
plt.savefig(f"{basefigdir}trans_inp{ext}", bbox_inches = mpl.transforms.Bbox([[1.7,0.80], [3.85,2.3]]), transparent = True)
plt.show()
plt.close()

# %%
