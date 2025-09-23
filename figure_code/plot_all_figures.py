"""Code for plotting all figure panels used in the STA paper"""

#%% packages
import pysta
from pysta import basedir
import os
os.chdir(f"{basedir}/figure_code")

#%% initialise directories
print("\nInitialising directories")
os.makedirs(f"{basedir}/figures", exist_ok = True)
os.makedirs(f"{basedir}/figures/schematics", exist_ok = True)

os.makedirs(f"{basedir}/figures/handcrafted_examples", exist_ok = True)
os.makedirs(f"{basedir}/figures/handcrafted_performance", exist_ok = True)

os.makedirs(f"{basedir}/figures/rnn_decoding", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_connectivity", exist_ok = True)
os.makedirs(f"{basedir}/figures/rnn_generalisation", exist_ok = True)
os.makedirs(f"{basedir}/figures/attractor", exist_ok = True)

os.makedirs(f"{basedir}/figures/changing_maze_rnn", exist_ok = True)

os.makedirs(f"{basedir}/figures/rnn_behaviour", exist_ok = True)
os.makedirs(f"{basedir}/figures/simple_tasks", exist_ok = True)
os.makedirs(f"{basedir}/figures/size", exist_ok = True)

#%% Figure 1
print("\nPlotting panels for figure 1")
import plot_fig1_schematics

#%% Figure 2
print("\nPlotting panels for figure 2")
import plot_fig2_schematics

#%% Figure 3
print("\nPlotting panels for figure 3")
import plot_handcrafted_examples
import plot_handcrafted_performance

#%% Figure 4
print("\nPlotting panels for figure 4")
import plot_fig4_schematics
import plot_rnn_decoding
import plot_rnn_generalization

#%% Figure 5
print("\nPlotting panels for figure 5")
import plot_fig5_schematics
import plot_rnn_connectivity
import plot_attractor_dynamics

#%% Figure 6
print("\nPlotting panels for figure 6")
import plot_fig6_schematics
import plot_changing_maze

#%% Supplementary figures
print("\nPlotting panels for supplementary figures")
import plot_rnn_performance
import plot_simple_tasks
import plot_by_size
import plot_supplementary_schematics

#%%
