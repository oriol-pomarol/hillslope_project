import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as tck
import matplotlib.colors as mc
from matplotlib.patches import Patch
import joblib as jb
from keras.models import load_model as keras_load_model
from config import plots as cfg
from config import paths

def equilibrium_plots(model_name='nn'):
  #Set the plot parameters
  B_lim = 3
  D_lim = 0.8
  colors = ['#b2df8a', '#55a868', '#a6cee3', '#4c72b0', 'yellow', 'red']
  labels = ['B - unstable equilibrium', 'B - stable equilibrium', 
            'D - unstable equilibrium', 'D - stable equilibrium', 
            'System - unstable equilibrium', 'System - stable equilibrium']

  # Make a grid of B and D values
  n_sq = int(B_lim * D_lim * 10) * cfg.scale_surface
  D_edges = np.linspace(0, D_lim, n_sq)
  B_edges = np.linspace(0, B_lim, n_sq)
  D_grid, B_grid = np.meshgrid(D_edges, B_edges)

  # Load the model
  model = load_model(model_name)
  
  # Load the equilibrium points if specified
  if cfg.load_splot_data:
    splot_data = jb.load(paths.outputs / 'splot_data.joblib')

  else:
    # Initialize the dictionary to store the streamplot data
    splot_data = {}

    # Add the missing data for the streamplot
    for g in cfg.g_stream:

      # Skip if the data is already stored
      if g in splot_data.keys():
        continue

      # Format the data to feed to the model
      X_pred = np.column_stack((B_grid.flatten(), D_grid.flatten(), np.full(n_sq**2, g)))

      # Use the model to predict the value of the derivatives in the grid points
      Y_pred = model.predict(X_pred).reshape((n_sq, n_sq, -1))

      # Find stable and unstable equilibrium points
      Y_eq = find_eq_points(Y_pred, cfg.thr_eq, B_grid, D_grid)

      # Find the feature space velocity and take the logarithm
      velocity = np.sqrt((Y_pred[:,:,0]/B_lim)**2 + (Y_pred[:,:,1]/D_lim)**2)
      log_vel = np.log10(velocity)

      # Store the results in the dictionary
      splot_data[g] = {
          'Y_pred': Y_pred,
          'Y_eq': Y_eq,
          'log_vel': log_vel
      }

    # Store the results in a joblib file
    jb.dump(splot_data, paths.outputs / 'splot_data.joblib')

  # Load the equilibrium points if specified
  if cfg.load_eq_points:
    eq_points = pd.read_csv(paths.outputs / 'eq_points.csv')

  else:
    # Find the system equilibrium for all g values, store streamplot if in g_plot
    eq_points = pd.DataFrame(columns=['B', 'D', 'g', 'type'])

    # Loop over the g values
    for g in np.linspace(0, 3, cfg.n_g_vals):
        
      # If in g_plot, use its results
      if g in cfg.g_stream.keys():
        Y_eq = splot_data[g]['Y_eq']

      else:
        # Format the data to feed to the model
        X_pred = np.column_stack((B_grid.flatten(), D_grid.flatten(),
                                np.full(n_sq**2, g)))

        # Use the model to predict the value of the derivatives in the grid points
        Y_pred = model.predict(X_pred).reshape((n_sq, n_sq, -1))

        # Find stable and unstable equilibrium points
        Y_eq = find_eq_points(Y_pred, cfg.thr_eq, B_grid, D_grid)

      # Unpack the results
      B_eq, D_eq = Y_eq['B'], Y_eq['D']

      # Find the equilibrium points of the whole system
      Y_eq_st = B_eq['stable'] & D_eq['stable']
      Y_eq_un = (B_eq['unstable'] & D_eq['unstable']) | \
              (B_eq['stable'] & D_eq['unstable']) | \
              (B_eq['unstable'] & D_eq['stable'])
      
      # Retrieve the B and D values for the equilibrium points
      B_eq_st, D_eq_st = B_grid[Y_eq_st], D_grid[Y_eq_st]
      B_eq_un, D_eq_un = B_grid[Y_eq_un], D_grid[Y_eq_un]

      # Store the results
      eq_points = pd.concat([eq_points, pd.DataFrame({'B': B_eq_st, 'D': D_eq_st,
                                                      'g': cfg.g_stream, 'type': 'stable'})],
                                                      ignore_index=True)
      eq_points = pd.concat([eq_points, pd.DataFrame({'B': B_eq_un, 'D': D_eq_un,
                                                      'g': g, 'type': 'unstable'})],
                                                      ignore_index=True)

    # Store the results in a csv file
    eq_points.to_csv(paths.outputs / 'eq_points.csv', index=False)

  # Plot the streamplots
  plot_streamplots(splot_data, B_lim, D_lim, B_grid, D_grid, colors, labels)

  # Plot the equilibrium points
  plot_eq_points(eq_points, colors, labels)

  # Create a summary of the results
  equilibrium_summary = "".join(['\n\n*EQUILIBRIUM PLOTS*',
                                 '\ng_plot = {}'.format(cfg.g_stream),
                                 '\nB_lim = {}'.format(B_lim),
                                 '\nD_lim = {}'.format(D_lim),
                                 '\nthr_eq = {}'.format(cfg.thr_eq),
                                 '\nn_g_vals = {}'.format(cfg.n_g_vals)])

  return equilibrium_summary

##############################################################################

def load_model(name):

  # If the model is a neural network, load it using keras
  if name == 'nn':
    model = keras_load_model(paths.models / 'nn_model.h5', compile=False)

  # If the model is a random forest, load it using joblib
  elif name == 'rf':
    model = jb.load(paths.models / 'rf_model.joblib')

  # If the model is the minimal model, define it
  elif name == 'mm':
    class MinimalModel:
      def predict(self, X):
        r, c, i, d, s = 2.1, 2.9, -0.7, 0.04, 0.4 
        Wo, a, Et, Eo, k, b, C = 5e-4, 4.02359478109, 0.021, 0.084, 0.05, 0.28, 1e-4
        
        B, D, g = X[:, 0], X[:, 1], X[:, 2]
        
        dB_dt_step = (1 - (1 - i) * np.exp(-1 * D / d)) * (r * B * (1 - B / c)) - g * B / (s + B)
        dD_dt_step = Wo * np.exp(-1 * a * D) - np.exp(-1 * B / b) * (Et + np.exp(-1 * D / k) * (Eo - Et)) - C
        
        return np.column_stack((dB_dt_step, dD_dt_step))
    model = MinimalModel()
  
  return model

##############################################################################

def find_eq_points(Y_pred, B_grid, D_grid):

  # Unpack the results
  dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]

  # Get the limits of the data
  B_lim, D_lim = np.max(B_grid), np.max(D_grid)

  # Find the equilibrium points given the threshold weighted by the std
  B_eq = np.abs(dB_dt) < cfg.thr_eq[0]
  D_eq = np.abs(dD_dt) < cfg.thr_eq[1]
  
  # Compute the gradients
  dB_dt_B, dB_dt_D = np.gradient(dB_dt)
  dD_dt_B, dD_dt_D = np.gradient(dD_dt)
  
  # Find the stable and unstable equilibrium points
  B_st_eq = B_eq & ((dB_dt_B < 0) | (B_grid < 1/len(B_grid) * B_lim)) #& (dB_dt_D < 0)
  B_un_eq = B_eq & ~B_st_eq

  D_st_eq = D_eq & ((dD_dt_D < 0) | (D_grid < 1/len(D_grid) * D_lim)) #& (dD_dt_B < 0 )
  D_un_eq = D_eq & ~D_st_eq

  # Store the results in a dictionary
  Y_eq = {
    'B': {'stable': B_st_eq, 'unstable': B_un_eq},
    'D': {'stable': D_st_eq, 'unstable': D_un_eq}
  }
  
  return Y_eq

##############################################################################

def plot_streamplots(splot_data, B_lim, D_lim, B_grid, D_grid, colors, labels):
  
  # Define the plot sizes
  fontsize_titles = 32
  fontsize_labels = 34
  fontsize_ticks = 30
  fontsize_legend = 26
  scatter_size = 50
  point_sizes = [scatter_size]*4 + [scatter_size*2] + [scatter_size*4]
 

  # Retrieve the number of subplots and the g values
  n_subplots = len(splot_data)

  # Create a grid of subplots with the default style
  plt.style.use('default')
  fig, axs = plt.subplots(1,n_subplots, figsize=(12*n_subplots + 3, 12), dpi=300)

  # If g_plot contains only one value, wrap axs in a list
  if not isinstance(axs, np.ndarray):
      axs = [axs]

  # Get the global min and max of log_vel across all g_plot
  global_log_vel_min = min(splot_data[g]['log_vel'].min() for g in cfg.g_stream)
  global_log_vel_max = max(splot_data[g]['log_vel'].max() for g in cfg.g_stream)

  # Create a Normalize object for the color mapping
  norm = mc.Normalize(vmin=global_log_vel_min, vmax=global_log_vel_max)
      
  for i, g in enumerate(cfg.g_stream):

    # Set the title
    axs[i].set_title(f'g = {g:.1f} kg/m$^2$/yr', fontsize=fontsize_titles, pad=20)

    # Retrieve the results from the dictionary
    Y_pred = splot_data[g]['Y_pred']
    Y_eq = splot_data[g]['Y_eq']
    log_vel = splot_data[g]['log_vel']

    # Unpack the results
    dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]
    B_eq, D_eq = Y_eq['B'], Y_eq['D']

    # Find the equilibrium points of the whole system
    Y_eq_st = B_eq['stable'] & D_eq['stable']
    Y_eq_un = (B_eq['unstable'] & D_eq['unstable']) | \
          (B_eq['stable'] & D_eq['unstable']) | \
          (B_eq['unstable'] & D_eq['stable'])

    # Set the axis tick parameters
    ax = axs[i]
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Make the streamplot with the normalized color mapping
    ax.streamplot(D_grid, B_grid, dD_dt, dB_dt, color=norm(log_vel), cmap=plt.cm.viridis,
                  minlength=0.01, linewidth=3, arrowsize=3, broken_streamlines=True, zorder=1)

    # Define the conditions and corresponding colors
    conditions = [B_eq['unstable'], B_eq['stable'],
            D_eq['unstable'], D_eq['stable'],
            Y_eq_un, Y_eq_st]

    # Iterate over conditions and colors and create scatter plots
    for condition, color, size in zip(conditions, colors, point_sizes):
      ax.scatter(D_grid[condition], B_grid[condition], color=color, s=size, zorder=3)

    ax.set_ylim(0, B_lim)
    ax.set_xlim(0, D_lim)
    if i == 0:
      ax.set_ylabel('Biomass ($kg/m^2$)', labelpad=20, fontsize=fontsize_labels)
    if i == 1:
      ax.set_xlabel('Soil depth ($m$)', labelpad=20, fontsize=fontsize_labels)
    
  # Create a legend
  legend_handles = [Patch(color=color, label=label) 
                    for color, label in zip(colors, labels)]
  plt.legend(handles=legend_handles, fontsize=fontsize_legend, loc='lower right')

  # Create a ScalarMappable with the same colormap and normalization
  sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax)
  cbar.set_label('Log reltive rate of change ($s^{-1}$)', labelpad=18, fontsize=fontsize_labels)

  # Save the figure
  plt.tight_layout()
  plt.savefig(paths.figures / 'streamplot_nullclines.png', dpi=300)

  return

##############################################################################

def plot_eq_points(eq_points, colors, labels):

  # Set the plot sizes
  fontsize_labels = 32
  fontsize_ticks = 30
  scatter_size = 200

  # Plot the equilibrium points
  plt.style.use('seaborn-v0_8')
  fig, axs = plt.subplots(2, figsize=(22, 13))

  # For biomass
  # Plot the unstable equilibrium points in light green
  axs[0].scatter(eq_points[eq_points['type'] == 'unstable']['g'],
                eq_points[eq_points['type'] == 'unstable']['B'],
                color=colors[0], label=labels[0],
                s=scatter_size)
  # Plot the stable equilibrium points in dark green
  axs[0].scatter(eq_points[eq_points['type'] == 'stable']['g'],
                eq_points[eq_points['type'] == 'stable']['B'],
                color=colors[1], label=labels[1],
                s=scatter_size)

  # For soil depth
  # Plot the unstable equilibrium points in light blue
  axs[1].scatter(eq_points[eq_points['type'] == 'unstable']['g'],
                eq_points[eq_points['type'] == 'unstable']['D'],
                color= colors[2], label=labels[2],
                s=scatter_size)
  # Plot the stable equilibrium points in dark blue
  axs[1].scatter(eq_points[eq_points['type'] == 'stable']['g'],
                eq_points[eq_points['type'] == 'stable']['D'],
                color=colors[3], label=labels[3],
                s=scatter_size)

  # Set the axis limits and labels
  axs[0].set_xlim([0, 3])
  axs[0].set_ylim(bottom=0)  # Start y-axis from 0
  axs[0].set_ylabel('Biomass ($kg/m^2$)', fontsize=fontsize_labels)

  axs[1].set_xlim([0, 3])
  axs[1].set_ylim(bottom=0)  # Start y-axis from 0
  axs[1].set_xlabel('Grazing pressure ($kg/m^2/yr$)', fontsize=fontsize_labels)
  axs[1].set_ylabel('Soil Depth ($m$)', fontsize=fontsize_labels)

  # Set the font size for tick labels
  axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
  axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

  # Reduce the number of ticks in the y axis for both plots
  axs[0].yaxis.set_major_locator(tck.MaxNLocator(nbins=6, prune='lower'))
  axs[1].yaxis.set_major_locator(tck.MaxNLocator(nbins=5, prune='lower'))

  # Add a legend
  axs[0].legend(loc='upper right', fontsize=fontsize_labels)
  axs[1].legend(loc='upper right', fontsize=fontsize_labels)

  # Save the figure
  plt.tight_layout()
  plt.savefig(paths.figures / 'eq_plot.png', dpi=300)

  return