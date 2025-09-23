"""Code for plotting handcrafted example representations"""

#%%

import pysta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from pysta import basedir
import copy
ext = ".pdf"
np.random.seed(0)

#%% set font with arial .ttf file
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = f"{basedir}/data/arial.ttf"
fm.fontManager.addfont(font_path)
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.size'] = 8

#%%
pysta.reload()
data = pickle.load(open(f"{basedir}/data/examples/example_task_data.pickle", "rb"))
cmap = "Greens"
vmin, vmax = 0.3, 3.2
no_loc = True
figsize, mouse_size, cheese_size = (3.30/2.54, 3.30/2.54), 320, 280

# plot moving goal
moving_goal = data["moving_goal"]
moving_goal2 = copy.deepcopy(moving_goal)
moving_goal2["optimal_actions"], moving_goal2["loc"] = moving_goal["optimal_actions_t2"], moving_goal["loc_t2"]
    
pysta.plot_utils.plot_flat_frame(filename = f"{basedir}/figures/handcrafted_examples/moving_goal1{ext}", vmap = moving_goal["rew"][0], **moving_goal, cmap = cmap, vmin = -0.6*vmin, vmax = +0.6*vmax, xlabel = None, show = True, figsize = figsize, mouse_size = mouse_size, cheese_size = cheese_size)
pysta.plot_utils.plot_flat_frame(filename = f"{basedir}/figures/handcrafted_examples/moving_goal2{ext}", vmap = moving_goal2["rew"][1], goal_step_num = 1, **moving_goal2, cmap = cmap, vmin = -0.6*vmin, vmax = +0.6*vmax, xlabel = None, show = True, figsize = figsize, mouse_size = mouse_size, cheese_size = cheese_size)

# plot static goal
static_goal = data["static_goal"]
pysta.plot_utils.plot_flat_frame(filename = f"{basedir}/figures/handcrafted_examples/static_goal{ext}", vmap = static_goal["rew"][0], **static_goal, cmap = cmap, vmin = -0.6*vmin, vmax = +0.6*vmax, show = True, figsize = figsize, mouse_size = mouse_size, cheese_size = cheese_size)

# plot second static goal
rew, goal, opt = static_goal["rew"][0] * 0, 2, static_goal["optimal_actions"]*0
rew[goal] += 1
opt[3] += 1

pysta.plot_utils.plot_flat_frame(filename = None, vmap = rew, optimal_actions = opt, walls = static_goal["walls"], loc = static_goal["loc"], goal = goal, cmap = cmap, vmin = -0.6*vmin, vmax = +0.6*vmax, show = False, figsize = figsize, mouse_size = mouse_size, cheese_size = cheese_size)
goal_loc = pysta.maze_utils.index_to_loc(10, 4)
plt.scatter(goal_loc[0]-0.05, goal_loc[1], color = np.ones(3)*0.45, marker = pysta.plot_utils.cheese_marker, s = cheese_size*0.75, zorder = 80, lw = 0.3)
plt.savefig(f"{basedir}/figures/handcrafted_examples/static_goal2{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()

#%%
vmin, vmax = 1.2,2.2
# plot reward landscape at different times
rew_land = data["rew_landscape"]
for i in range(4):
    rew_land_i = copy.deepcopy(rew_land)
    rew_land_i["loc"] = rew_land["all_locs"][i]
    rew_land_i["optimal_actions"] = None if i >= 2 else rew_land["optimal_actions"]
    pysta.plot_utils.plot_flat_frame(filename = f"{basedir}/figures/handcrafted_examples/rew_land{i+1}{ext}", vmap = rew_land["rews"][i], **rew_land_i, cmap = cmap, vmin = -1*vmin, vmax = +1*vmax, xlabel = None, show = True, figsize = figsize, mouse_size = mouse_size, cheese_size = cheese_size)

#%%
plt.figure(figsize = (0.6,4))
plt.imshow(np.linspace(+1, -1, 101)[:, None], vmin = -1*vmin, vmax = vmax, aspect = "auto", cmap = cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(f"{basedir}/figures/handcrafted_examples/rew_cbar{ext}", bbox_inches = "tight", transparent = True)
plt.show()
plt.close()


#%% now plot example representations

figsize, mouse_size, cheese_size = (3.65/2.54, 3.65/2.54), 340, 300

data = pickle.load(open(f"{basedir}/data/examples/example_rep_data.pickle", "rb"))
env_labels = ["static", "changing", "moving", "landscape"]
env_titles = ["constant goal", "changing goal", "moving goal", "reward landscape"]

for agent in ["td", "sr", "sta"]:
    for env in env_labels:
        kwargs = data[env][agent]
        fname = f"{basedir}/figures/handcrafted_examples/{env}_{agent}{ext}"
        
        if agent == "sta":
            kwargs["vmap"] = kwargs["vmap"][:-1, :]
            pysta.plot_utils.plot_perspective_attractor(filename = fname, **kwargs, cmap = "YlOrRd", vmin = -0.15, vmax = 1.3, show = True, goal_inds = range((0 if env == "moving" else 1), kwargs["vmap"].shape[0]), bbox_inches = mpl.transforms.Bbox([[2.5,1.6], [5.9,2.8]]), figsize = (8.22,4.4))
        else:
            if agent == "td":
                vmin, vmax = (0.45*0.85, 1.3) if env == "static" else (0.541, 0.6)
            if agent == "sr":
                vmin, vmax = 0.2, 5.0
            pysta.plot_utils.plot_flat_frame(filename = fname, **kwargs, cmap = "YlOrRd", vmin = vmin, vmax = vmax, show = True, figsize = figsize, mouse_size = mouse_size, cheese_size = cheese_size)


# %%
