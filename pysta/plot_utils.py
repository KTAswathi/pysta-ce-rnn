import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import subprocess

import matplotlib as mpl
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch

from .maze_utils import index_to_loc, action_deltas, compute_adjacency
from .utils import basedir
from scipy.ndimage import gaussian_filter1d

base_maze_col = np.ones(3)*0.7

def process_marker(marker):
    # center marker
    marker.vertices = marker.vertices - 0.5*marker.vertices.min(0) - 0.5 * marker.vertices.max(0)
    # rotate 180
    marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    # flip x axis
    marker = marker.transformed(mpl.transforms.Affine2D().scale(-1,1))
    return marker

_, cheese_attributes = svg2paths(f"{basedir}/pysta/files/cheese.svg")
cheese_path = "".join([att["d"] for att in cheese_attributes])
cheese_marker = process_marker(parse_path(cheese_path))

_, mouse_attributes = svg2paths(f"{basedir}/pysta/files/mouse.svg")
mouse_path = "".join([att["d"] for att in mouse_attributes])
mouse_marker = process_marker(parse_path(mouse_path))


def plot_flat_frame(walls, figsize = (1.5,1.5), filename = None, cheese_size = 350, mouse_size = 450, lw = 4.5, goal = None, loc = None, optimal_actions = None, ax = None, vmap = None, goal_step_num = 0, cmap = "coolwarm", vmin = None, vmax = None, xlabel = None, show = False, **kwargs):
    """
    Function for plotting a snapshot of an environment
    
    Parameters
    ----------
    filename : str
        name of file to save to
    goal : int or tensor
        index of goal location. optionally full trajectory (don't plot if None)
    loc : int
        index of current location (don't plot if None)
    optimal_actions : list
        set of optimal actions as one-hot vectors
    ax : plt.axis
        if provided, plot on this ax instead of creating a new one
    vmap : tensor
        values to plot as a heatmap
    cmap : str
        plt colormap to use
    """
    
    if ax is None:
        plt.figure(figsize = figsize)
        ax = plt.gca()
    
    N = walls.shape[0]
    L = int(np.sqrt(N))

    ax.imshow(np.zeros((L, L)), cmap = "coolwarm", vmin = -1, vmax = 1, zorder = -100) # plot base environment
    # plot base grid of the world
    for i in range(L+1):
        ax.axvline(i-0.5, color = "k", zorder = 1)
        ax.axhline(i-0.5, color = "k", zorder = 1)

    for s in range(N): #for each state
        for i in range(walls.shape[-1]): #for each neighbor
            if bool(walls[s, i]):
                state = index_to_loc(s, L)
                if i == 0: #wall to the right
                    z1, z2 = state + np.array([0.5, 0.5]), state + np.array([0.5, -0.5])
                elif i == 1: #wall to the left
                    z1, z2 = state + np.array([-0.5, 0.5]), state + np.array([-0.5, -0.5])
                elif i == 2: #wall above
                    z1, z2 = state + np.array([0.5, 0.5]), state + np.array([-0.5, 0.5])
                elif i == 3: #wall below
                    z1, z2 = state + np.array([0.5, -0.5]), state + np.array([-0.5, -0.5])

                ax.plot([z1[0], z2[0]], [z1[1], z2[1]], color="k", ls="-", lw=lw)

    if vmap is not None: # optionally overlay a value map
        if vmin is None:
            vmin = vmap.min()
        if vmax is None:
            vmax = vmap.max()
        ax.imshow(vmap.reshape(L, L).T, cmap = cmap, vmin = vmin, vmax = vmax, zorder = -50)
    
    # plot goal location/trajectory
    if goal == None:
        None # don't plot goal
    elif type(goal) == int:
        goal_loc = index_to_loc(goal, L)
        ax.scatter(goal_loc[0]-0.04, goal_loc[1], color = "k", marker = cheese_marker, s = cheese_size, zorder = 80, lw = 0.75)
    else:
        goal_loc = index_to_loc(goal, L)[:, goal_step_num:]
        for i_g, g in enumerate(goal_loc.T[:-1]):
            g1 = goal_loc[:, i_g+1]
            col, alpha = np.ones(3)*0.5, 1.0
            col = np.array([222, 155, 0])/255
            col = np.ones(3)*0.5
            arrow = (i_g == goal_loc.shape[-1]-2)
            ax.arrow(g[0], g[1], g1[0]-g[0], g1[1]-g[1], color = col, alpha = alpha, length_includes_head = True, width = 0.1, head_width = 0.35*arrow, head_length = 0.45*arrow)
        ax.scatter(goal_loc[0, 0]-0.04, goal_loc[1, 0], color = "k", marker = cheese_marker, s = cheese_size, zorder = 99, lw = 0.75)
    
    if loc is not None: # plot current location
        agent_loc = index_to_loc(loc, L)
        ax.scatter(agent_loc[0], agent_loc[1], color = "k", marker = mouse_marker, s = mouse_size, lw = 1.25, zorder = 100)

    
    if optimal_actions is not None:
        opt_col = np.array([222, 155, 0])/255
        opt_col = np.array([245, 4, 0])/255
        optimal_action_inds = torch.where(optimal_actions)[0] # turn 1hot into indices
        if len(optimal_actions) == N: # allocentric actions
            for action in optimal_action_inds:
                loc = index_to_loc(action, L)
                ax.scatter(loc[0], loc[1], color = opt_col, marker = ".", s = 90, lw = 2.5, zorder = 90)
        else: # egocentric
            for action in optimal_action_inds:
                dloc = np.array(action_deltas[action])
                base = agent_loc + dloc*0.4
                ax.arrow(base[0], base[1], dloc[0]*0.7, dloc[1]*0.7, color = opt_col, width = 0.1, head_width = 0.35, head_length = 0.3, length_includes_head = True, zorder = 50)

    # parameters and save/show
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.5, L-0.5)
    ax.set_ylim(-0.5, L-0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight", transparent = True)
        
    if show:
        plt.show()
    
    if (filename is not None) or show:
        plt.close()


def plot_maze_scaffold(adjacency, ax = None, zloc = None, cols = None, maze_col = None, s = 750, lw = 8, z = None, vmap = None, cmap = "YlOrRd", edgecols = None):
    """function for plotting a single maze"""
    
    maze_col = base_maze_col if maze_col is None else maze_col
    N = adjacency.shape[0]
    L = int(np.sqrt(N))

    if ax is None: ax = plt.gca()

    xs, ys = [np.array([index_to_loc(i, L = L)[j] for i in range(N)]) for j in [0,1]]

    for n1, n2 in zip(*np.where(adjacency == 1)):
        if (n1 != n2):
            (n1x, n1y), (n2x, n2y) = [index_to_loc(n, L = L) for n in [n1, n2]]
            if max(n1x, n1y, n2x, n2y) <= L-1:
                ax.plot([n1x, n2x], [n1y, n2y], color = maze_col, lw = lw)
                
    ax.scatter(xs, ys, color = maze_col, s = s, marker = "8", zorder = 1)
    
    if vmap is not None:
        ax.scatter(xs, ys, c = vmap, cmap = cmap, marker = ".", zorder = 2, s = s*1.8, edgecolors = edgecols, lw = 2.5)

    ax.set_xlim(-0.5, L-0.5)
    ax.set_ylim(-0.5, L-0.5)
    return

#%% code for plotting STA representations

def plot_perspective_attractor(walls, vmap, state_actions = False, maze_col = None, act_cols = None, override_cols = [], loc = None, goal = None, vmin = None,
                               vmax = None, dpi = 100, cmap = "YlOrRd", filename = None, lw = 4, plot_proj = True, figsize = (7,4),
                               aspect = (1,1,4.5), view_init = (-30,-10,-90), plot_subs = True, extra_poly = None, circ_r = 0.22, goal_inds = None,
                               show = False, bbox_inches = "tight", edgecolors = None, transparent = True):
    """
    
    Parameters
    -----------
    vmap : array
        value to imshow at each time and state. Size: (num_mod, num_locs)
    """
    
    maze_col = base_maze_col if maze_col is None else maze_col
    
    if vmin is None:
        vmin = vmap.min()
    if vmax is None:
        vmax = vmap.max()

    vmap = (vmap - vmin)/(vmax - vmin + 1e-20)
    
    Nmod, N = vmap.shape[:2]
    L = int(np.sqrt(N))
    adj = compute_adjacency(walls)[0]

    oct_deltas = [np.array([np.cos(theta), np.sin(theta), 0])*0.37 for theta in np.linspace(2*np.pi/16,2*np.pi*(1+1/16), 9)[:-1]]
    circ_deltas = [np.array([np.cos(theta), np.sin(theta), 0])*circ_r for theta in np.linspace(0,2*np.pi, 25)[:-1]]
    square_deltas = [np.array([np.cos(theta), np.sin(theta), 0])*np.sqrt(2)/2 for theta in np.linspace(0,2*np.pi, 5)[:-1]+np.pi/4]

    zeff = Nmod + 0.3

    verts, verts2, verts3, cols, cols2, cols3 = [[] for _ in range(6)]
    if edgecolors is not None: verts4 = []
    
    thicklines, thickcols = [], []
    for z in range(Nmod):
        act = vmap[z, ...]

        for n1, n2 in zip(*np.where(np.tril(adj, -1) == 1)):
            (n1x, n1y), (n2x, n2y) = [index_to_loc(n, L = L) for n in [n1, n2]]
            n1x, n2x, n1y, n2y = min(n1x, n2x), max(n1x, n2x), L-1-min(n1y, n2y), L-1-max(n1y, n2y)

            cols.append(maze_col)
            width = 0.08
            eps = 0.00
            if n1x == n2x: # horizontal
                verts.append([(n1x-width, n1y, z+eps), (n1x+width, n1y, z+eps), (n1x+width, n2y, z+eps), (n1x-width, n2y, z+eps)])
            else: # vertical
                verts.append([(n1x, n1y-width, z+eps), (n1x, n1y+width, z+eps), (n2x, n1y+width, z+eps), (n2x, n1y-width, z+eps)])

        for izloc, zloc in enumerate(index_to_loc(np.arange(N), L = L).T): # add octagons
            zloc = np.array([zloc[0], L-1-zloc[1], z])
            verts2.append([zloc+delta for delta in oct_deltas])
            cols2.append(maze_col)
            
            if state_actions: # plot for every state_action
                offsets = [np.array(offset) for offset in [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)]]
                for offset in offsets:
                    verts3.append([zloc+0.53*delta+0.21*offset for delta in circ_deltas])
                    if edgecolors is not None:
                        verts4.append([zloc+0.5*delta*1.2+0.22*offset for delta in circ_deltas])
            else:
                verts3.append([zloc+delta+np.array([0,0,0]) for delta in circ_deltas])
                if edgecolors is not None:
                    verts4.append([zloc+delta*1.3+np.array([0,0,0]) for delta in circ_deltas])
            
            if state_actions:
                for itrans in range(4):
                    if act_cols is None:
                        cols3.append(plt.get_cmap(cmap)(act[izloc][itrans]))
                    else:
                        cols3.append(act_cols[z][izloc][itrans])
            else:
                if act_cols is None: # apply cmap to scalar values
                    cols3.append(plt.get_cmap(cmap)(act[izloc]))
                else: # directly use colors provided
                    cols3.append(act_cols[z][izloc])


    for override in override_cols:
        ind = override[0]*N+override[1]
        cols3[ind] = override[2]

    # also plot a max projection
    if type(vmap) == np.ndarray:
        maxact = vmap.max(axis = 0)
    elif type(vmap) == torch.Tensor:
        maxact = vmap.amax(axis = 0)
    else:
        raise NotImplementedError
        
    maxverts, maxcols = [], []
    for izloc, zloc in enumerate(index_to_loc(np.arange(N), L = L).T): # 
        zloc = np.array([zloc[0], L-1-zloc[1], zeff])
        maxverts.append([zloc+delta for delta in square_deltas])
        maxcols.append(plt.get_cmap(cmap)(maxact[izloc]))
    maxlines, maxlinecols = [], []
    for xy in range(L+1):
        eps = 0.001
        xy = xy - 0.5
        maxlines.append([(xy-eps, -0.5,zeff), (xy+eps, -0.5,zeff), (xy+eps, L-0.5, zeff), (xy-eps, L-0.5, zeff)])
        maxlines.append([(-0.5, xy-eps,zeff), (-0.5, xy+eps,zeff), (L-0.5, xy+eps, zeff), (L-0.5, xy-eps, zeff)])
        maxlinecols.append(np.zeros(3))
        maxlinecols.append(np.zeros(3))

    eps = 0.12
    for n1 in range(N):
        for n2 in range(n1, N):
            (n1x, n1y), (n2x, n2y) = [index_to_loc(n, L) for n in [n1, n2]]
            n1x, n2x, n1y, n2y = min(n1x, n2x), max(n1x, n2x), L-1-min(n1y, n2y), L-1-max(n1y, n2y)
            if (np.abs(n1x-n2x)+np.abs(n1y-n2y) == 1) and (adj[n1, n2] == 0): # adjacent but not connected

                thickcols.append(np.zeros(3))
                if n1x == n2x: # horizontal
                    x1, x2, y1, y2 = n1x-0.5, n1x+0.5, 0.5*(n1y+n2y)-eps, 0.5*(n1y+n2y)+eps
                    thicklines.append([(x1, y1, zeff), (x1, y2, zeff), (x2, y2, zeff), (x2, y1, zeff)])
                else: # vertical
                    y1, y2, x1, x2 = n1y-0.5, n1y+0.5, 0.5*(n1x+n2x)-eps, 0.5*(n1x+n2x)+eps
                    thicklines.append([(x1, y1, zeff), (x1, y2, zeff), (x2, y2, zeff), (x2, y1, zeff)])
    thicklines.append([(-0.5-eps,-0.5,zeff), (-0.5-eps,L-0.5,zeff),(-0.5+eps,L-0.5,zeff), (-0.5+eps,-0.5,zeff)])
    thicklines.append([(L-0.5-eps,-0.5,zeff), (L-0.5-eps,L-0.5,zeff),(L-0.5+eps,L-0.5,zeff), (L-0.5+eps,-0.5,zeff)])

    thicklines.append([(-0.5,-0.5-eps,zeff), (L-0.5,-0.5-eps,zeff),(L-0.5,-0.5+eps,zeff), (-0.5,-0.5+eps,zeff)])
    thicklines.append([(-0.5,L-0.5-eps,zeff), (L-0.5,L-0.5-eps,zeff),(L-0.5,L-0.5+eps,zeff), (-0.5,L-0.5+eps,zeff)])
    for _ in range(4): thickcols.append(np.zeros(3))


    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(projection='3d', computed_zorder = False)

    if plot_proj:
        ax.add_collection3d(Poly3DCollection(maxverts,color=maxcols))
        ax.add_collection3d(Poly3DCollection(maxlines,color=maxlinecols, lw = 0.2))
        ax.add_collection3d(Poly3DCollection(thicklines,color=thickcols, lw = 1e-3))

    if plot_subs: # plot subspaces
        lims = (-0.4, L-1+0.4)
        verts_sub = [[(lims[0],lims[0],z),(lims[0],lims[1],z),(lims[1],lims[1],z),(lims[1],lims[0],z)] for z in range(Nmod)]
        cols_sub = [(0.5,0.5,0.5,0.15) for z in range(Nmod)]
        ax.add_collection3d(Poly3DCollection(verts_sub,color=cols_sub))
        
        
    ax.add_collection3d(Poly3DCollection(verts,color=cols))
    ax.add_collection3d(Poly3DCollection(verts2,color=cols2))
    
    if extra_poly is not None:
        ax.add_collection3d(extra_poly)
    
    ax.add_collection3d(Poly3DCollection(verts3,color=cols3))
    
    if edgecolors is not None:
        ax.add_collection3d(Poly3DCollection(verts4, color = (1,1,1,0), edgecolor = edgecolors, linewidth = 1.15))

    # add goal and agent locations
    if goal is not None:
        goal_inds = list(range(0, z+1)) if goal_inds is None else list(goal_inds)
        if type(goal) in [int, np.int64]:
            goal_loc = index_to_loc(goal, L)[:, None]+np.zeros((2, Nmod+1))
            zgoals = goal_inds+[zeff] # also plot goal in projection plane
            goal_inds.append(goal_inds[-1]+1)
        else:
            goal_loc = index_to_loc(goal, L)
            zgoals = goal_inds
        
        for iz, zval in enumerate(zgoals):
            ax.scatter(goal_loc[0, goal_inds[iz]], L-1-goal_loc[1, goal_inds[iz]], zval, color = "k", marker = cheese_marker, s = 60, lw = 0.3)
            
    if loc is not None:
        if (type(loc) == int) or (len(loc.shape) == 0) or (len(loc.shape) == 1 and loc.shape[0] == 1): # if given as an index
            loc = index_to_loc(loc, L) # convert to (x,y)
        if plot_proj:
            ax.scatter(loc[0], L-1-loc[1], zeff, color = "k", marker = mouse_marker, s =115, lw = 0.6)
        ax.scatter(loc[0], L-1-loc[1], 0, color = "k", marker = mouse_marker, s = 115, lw = 0.6)

    ax.set_box_aspect(aspect = aspect)

    ax.axis("off")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(view_init[0], view_init[1], view_init[2])

    if plot_proj:
        ax.set_zlim(-0.1, zeff + 0.2)
    else:
        ax.set_zlim(-0.1, z + 0.2)
    ax.set_xlim(-0.1, L-1+0.1)
    ax.set_ylim(-0.1, L-1+0.1)

    if filename is not None:
        plt.savefig(filename, bbox_inches = bbox_inches, transparent = transparent, dpi = dpi)

    if show:
        plt.show()
    
    if show or (filename is not None):
        plt.close()
        return
    
    return ax
                    

def plot_perspective_gif(walls, vmaps, tempdir = "./temp/", locs = None, filename = "test", delay = 50, print_freq = 10, delete_images_after = True, transparent = False, smooth = 5, vmin = None, vmax = None, **kwargs):
    """
    
    Parameters
    -----------
    vmaps : array
        value to imshow at each time and state. Size: (time, num_mod, num_locs)
    """
    
    os.makedirs(tempdir, exist_ok = True)
    N = walls.shape[0]
    L = int(np.sqrt(N))

    if locs is None:
        locs = [None for _ in range(len(vmaps))]
    else:
        locs = index_to_loc(locs, L = L).T.astype(float)
        locs = gaussian_filter1d(locs, smooth, axis = 0, mode = "nearest") if smooth > 0 else locs
    vmaps = gaussian_filter1d(vmaps, smooth, axis = 0, mode = "nearest") if smooth > 0 else vmaps

    nplot = len(vmaps)
    
    if vmin is None: vmin = np.amin(vmaps)
    if vmax is None: vmax = np.amin(vmaps)

    for iact, vmap in enumerate(vmaps):
        if iact % print_freq == 0:
            print("Plotting image", iact, "of", nplot, locs[iact])
        iname = f"{tempdir}/n{str(iact).zfill(3)}.png"
        plot_perspective_attractor(walls, vmap, vmin = vmin, vmax = vmax, loc = locs[iact], filename = iname, transparent = transparent, **kwargs)
        plt.close()

    print("converting to gif:", filename)
    subprocess.run(f"convert -delay {delay} {tempdir}/*.png {filename}", shell = True)
    if delete_images_after:
        subprocess.run(f"rm {tempdir}/*.png", shell = True)
    return


#%% code for analysis results

def plot_prediction_result(prediction_result, neural_times, loc_times, error = None, figsize = (3.0,2.5), labelpad = None, labels = None, filename = None, show = False, ymax = None, baseline = None, ts_train = None, legend = False, xlabel = None, cols = None, xticks = None, yticks = None):
    """
    function for plotting the result of predicting future/past locations from neural activity
    
    Parameters
    ----------
    prediction_result : array
        NxM array of performance when predicting location at one of M time points from neural activity at one of N.
    neural_times : list
        list of the times from which neural activity is used
    loc_times : list
        list of the times at which location is predicted
    filename : str
        name of file to save to
    show : bool
        whether to show the plot
    """
    
    plt.figure(figsize = figsize)
    
    xs = np.arange(prediction_result.shape[-1]) if loc_times is None else loc_times
    if neural_times is None:
        neural_times = np.arange(prediction_result.shape[0])
    
    if cols is None: 
        cols = [plt.get_cmap("tab10")(i) for i in range(len(prediction_result))]
        
    for i in range(0,len(prediction_result)-0):
        mean_pred = prediction_result[i]
        label = f"neurons from t = {neural_times[i]}" if labels is None else labels[i]
        plt.plot(xs, mean_pred, label = label, lw = 2, color = cols[i])
        if error is not None:
            std_pred = error[i]
            plt.fill_between(xs, mean_pred-std_pred, mean_pred+std_pred, alpha = 0.2, color = cols[i])
            
    plt.xlim(xs[0], xs[-1])

    if baseline is not None:
        plt.axhline(baseline, color = "k")
    
    if ts_train is not None:
        itrain_x, itrain_y = np.where(neural_times == ts_train[0])[0][0], np.where(loc_times == ts_train[1])[0][0]
        ytrain = prediction_result[itrain_x, itrain_y]
        plt.scatter([itrain_y], [ytrain], edgecolors = cols[itrain_x], facecolors = 'none', marker = "o", s = 120, lw = 2.5, zorder = 100)
    
    xlabel = "predict location at this time" if xlabel is None else xlabel
    
    plt.xlabel(xlabel)
    plt.ylabel("accuracy", labelpad = labelpad)
    
    if xticks is None:
        xticks = xs
    plt.xticks(xticks)
    
    if ymax is None:
        ymax = np.ceil(np.amax(prediction_result)*5)/5
    plt.ylim(0, ymax)
    
    if yticks is None:
        yticks = [tick for tick in [0,0.2,0.4,0.6,0.8,1] if tick <= ymax]
    plt.yticks(yticks)
    
    if legend:
        plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.3), ncol = 2, frameon = False, fontsize = 8)
    plt.gca().spines[['right', 'top']].set_visible(False)
    
    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight", transparent = True)
    if show:
        plt.show()
    if show or filename:
        plt.close()
    
    return


def plot_slot_connectivity(W, num_locs, vmin = 0.01, vmax = 0.995, filename = None, show = True, title = None, xticks = None, yticks = None, xlabel = "input", ylabel = "output", figsize = (4,4), xtickrot = 45, transparent = True):
    """
    For some effective weight matrix W, plot a heatmap of the connectivity.
    
    Parameters:
    ----------
    W : np.array
        weight matrix to plot
    num_locs : int
        number of locations in the environment
    """
    plt.figure(figsize = figsize)
    plt.imshow(W, cmap = "coolwarm", vmin = np.quantile(W, vmin), vmax = np.quantile(W, vmax), interpolation = 'none')
    nslots_y, nslots_x = [int(shape / num_locs) for shape in W.shape]
    for i in range(nslots_y-1):
        plt.axhline(num_locs*(i+1)-0.5, color = "k")
    for i in range(nslots_x-1):
        plt.axvline(num_locs*(i+1)-0.5, color = "k")
    
    if xticks is None:
        xticklabs = [f"subspace {i}" for i in range(nslots_x)]
    else:
        xticklabs = xticks
    if yticks is None:
        yticklabs = [f"subspace {i}" for i in range(nslots_y)]
    else:
        yticklabs = yticks
        
    xticklocs = [(i+0.5)*num_locs for i in range(len(xticklabs))]
    yticklocs = [(i+0.5)*num_locs for i in range(len(yticklabs))]
    
    plt.xticks(xticklocs, xticklabs, rotation = xtickrot, ha = ("right" if xtickrot > 0 else "center"))
    plt.yticks(yticklocs, yticklabs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    
    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight", transparent = transparent)
    if show:
        plt.show()
        
    if filename or show:
        plt.close()
        return
    else:
        return plt.gca()

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)
