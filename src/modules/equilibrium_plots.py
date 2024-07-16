import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import matplotlib.ticker as tck
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mc
import joblib as jb
from keras.models import load_model as keras_load_model
from config import paths

def equilibrium_plots(model_name='nn'):
  #Set the plot parameters
  g_plot = [0, 1.5, 3.0]
  scale_surface = 10
  B_lim = 3
  D_lim = 0.8
  threshold_eq = 1e-3, 1e-6
  n_g_values = 10

  # Make a grid of B and D values
  n_sq = int(B_lim * D_lim * 10) * scale_surface
  D_edges = np.linspace(0, D_lim, n_sq)
  B_edges = np.linspace(0, B_lim, n_sq)
  D_grid, B_grid = np.meshgrid(D_edges, B_edges)

  # Load the model
  model = load_model(model_name)

  # Initialize a dictionary to store the results for each grazing pressure
  results = {}

  for g in g_plot:
      # Format the data to feed to the model
      X_pred = np.column_stack((B_grid.flatten(), D_grid.flatten(), np.full(n_sq**2, g)))

      # Use the model to predict the value of the derivatives in the grid points
      Y_pred = model.predict(X_pred).reshape((n_sq, n_sq, -1))

      # Find stable and unstable equilibrium points
      Y_eq = find_eq_points(Y_pred, threshold_eq, B_grid, D_grid)

      # Find the feature space velocity and take the logarithm
      velocity = np.sqrt((Y_pred[:,:,0]/B_lim)**2 + (Y_pred[:,:,1]/D_lim)**2)
      log_vel = np.log10(velocity)

      # Store the results in the dictionary
      results[g] = {
          'Y_pred': Y_pred,
          'Y_eq': Y_eq,
          'log_vel': log_vel
      }

  # Set the default font size
  plt.rcParams['font.size'] = 32

  # Set the font size for the labels, title, and ticks
  plt.rcParams['axes.labelsize'] = 34
  plt.rcParams['xtick.labelsize'] = 30
  plt.rcParams['ytick.labelsize'] = 30

  # Create a grid of subplots
  fig, axs = plt.subplots(1,len(g_plot), figsize=(12*len(g_plot) + 3, 12), dpi=300)

  # If g_plot contains only one value, wrap axs in a list
  if not isinstance(axs, np.ndarray):
      axs = [axs]

  # Get the global min and max of log_vel across all g_plot
  global_log_vel_min = min(results[g]['log_vel'].min() for g in g_plot)
  global_log_vel_max = max(results[g]['log_vel'].max() for g in g_plot)

  # Create a Normalize object for the color mapping
  norm = mc.Normalize(vmin=global_log_vel_min, vmax=global_log_vel_max)
      
  for i, g in enumerate(g_plot):

    # Set the title
    axs[i].set_title(f'g = {g:.1f} kg/m$^2$/yr', fontsize=32, pad=20)

    # Retrieve the results from the dictionary
    Y_pred = results[g]['Y_pred']
    Y_eq = results[g]['Y_eq']
    log_vel = results[g]['log_vel']

    # Unpack the results
    dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]
    B_eq, D_eq = Y_eq['B'], Y_eq['D']

    # Find the equilibrium points of the whole system
    Y_eq_st = B_eq['stable'] & D_eq['stable']
    Y_eq_un = (B_eq['unstable'] & D_eq['unstable']) | \
          (B_eq['stable'] & D_eq['unstable']) | \
          (B_eq['unstable'] & D_eq['stable'])

    # Plot the equilibrium lines
    ax = axs[i]
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))

    # Make the streamplot with the normalized color mapping
    stream = ax.streamplot(D_grid, B_grid, dD_dt, dB_dt, color=norm(log_vel), cmap=plt.cm.viridis,
      minlength=0.01, linewidth=3, arrowsize=3, broken_streamlines=True, zorder=1)

    # Define the conditions and corresponding colors
    conditions = [B_eq['unstable'], B_eq['stable'],
            D_eq['unstable'], D_eq['stable'],
            Y_eq_un, Y_eq_st]
    colors = ['lightgreen', 'green', 'lightblue', 'blue', 'yellow', 'red']
    point_sizes = [50, 50, 50, 50, 100, 200]

    labels = ['B - unstable equilibrium', 'B - stable equilibrium', 
            'D - unstable equilibrium', 'D - stable equilibrium', 
            'System - unstable equilibrium', 'System - stable equilibrium']


    # Iterate over conditions and colors and create scatter plots
    for condition, color, size in zip(conditions, colors, point_sizes):
      ax.scatter(D_grid[condition], B_grid[condition], color=color, s=size, zorder=3)

    ax.set_ylim(0, B_lim)
    ax.set_xlim(0, D_lim)
    if i == 0:
      ax.set_ylabel('Biomass ($kg/m^2$)', labelpad=20)
    if i == 1:
      ax.set_xlabel('Soil depth ($m$)', labelpad=20)
    
  # Create a legend
  from matplotlib.patches import Patch
  legend_handles = [Patch(color=color, label=label) 
                    for color, label in zip(colors, labels)]
  plt.legend(handles=legend_handles, fontsize=26, loc='lower right')

  # Create a ScalarMappable with the same colormap and normalization
  sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
  sm.set_array([])  # You can set an empty array
  cbar = fig.colorbar(sm, ax=ax)
  cbar.set_label('Log reltive rate of change ($s^{-1}$)', labelpad=18)
  plt.tight_layout()
  plt.savefig(paths.figures / 'streamplot_nullclines.png', dpi=300)

  # Make a grid of B and D values
  D_edges = np.linspace(0, D_lim, n_sq)
  B_edges = np.linspace(0, B_lim, n_sq)
  D_grid, B_grid = np.meshgrid(D_edges, B_edges)

  # Load the model
  model = load_model('nn')

  # Find the system equilibrium for all g values
  eq_points = pd.DataFrame(columns=['B', 'D', 'g', 'type'])

  # Loop over the g values
  for g_plot in np.linspace(0, 3, n_g_values):

      # Format the data to feed to the model
      X_pred = np.column_stack((B_grid.flatten(), D_grid.flatten(),
                              np.full(n_sq**2, g_plot)))

      # Use the model to predict the value of the derivatives in the grid points
      Y_pred = model.predict(X_pred).reshape((n_sq, n_sq, -1))

      # Find stable and unstable equilibrium points
      Y_eq = find_eq_points(Y_pred, threshold_eq, B_grid, D_grid)

      # Unpack the results
      dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]
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
                                                      'g': g_plot, 'type': 'stable'})],
                                                      ignore_index=True)
      eq_points = pd.concat([eq_points, pd.DataFrame({'B': B_eq_un, 'D': D_eq_un,
                                                      'g': g_plot, 'type': 'unstable'})],
                                                      ignore_index=True)

  # Plotting the equilibrium points
  fig, axs = plt.subplots(2, figsize=(22, 13))

  # For biomass (x component)
  # Plot the unstable equilibrium points in light green
  axs[0].scatter(eq_points[eq_points['type'] == 'unstable']['g'],
                eq_points[eq_points['type'] == 'unstable']['B'],
                color='lightgreen', label='Unstable B equilibria')
  # Plot the stable equilibrium points in dark green
  axs[0].scatter(eq_points[eq_points['type'] == 'stable']['g'],
                eq_points[eq_points['type'] == 'stable']['B'],
                color='g', label='Stable B equilibria')

  # For soil depth (y component)
  # Plot the unstable equilibrium points in light blue
  axs[1].scatter(eq_points[eq_points['type'] == 'unstable']['g'],
                eq_points[eq_points['type'] == 'unstable']['D'],
                color='lightblue', label='Unstable D equilibria')
  # Plot the stable equilibrium points in dark blue
  axs[1].scatter(eq_points[eq_points['type'] == 'stable']['g'],
                eq_points[eq_points['type'] == 'stable']['D'],
                color='b', label='Stable D equilibria')

  # Set the axis limits and labels
  axs[0].set_xlim([0, 3])
  axs[0].set_ylim(bottom=0)  # Start y-axis from 0
  axs[0].set_ylabel('Biomass ($kg/m^2$)')

  axs[1].set_xlim([0, 3])
  axs[1].set_ylim(bottom=0)  # Start y-axis from 0
  axs[1].set_xlabel('Grazing pressure ($kg/m^2/yr$)')
  axs[1].set_ylabel('Soil Depth ($m$)')

  # Add a legend
  axs[0].legend()
  axs[1].legend()

  plt.tight_layout()
  plt.savefig(paths.figures / 'eq_plot.png', dpi=300)

  return ''

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

def find_eq_points(Y_pred, threshold_eq, B_grid, D_grid):

  # Unpack the results
  dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]

  # Get the limits of the data
  B_lim, D_lim = np.max(B_grid), np.max(D_grid)

  # Find the equilibrium points given the threshold weighted by the std
  B_eq = np.abs(dB_dt) < threshold_eq[0]
  D_eq = np.abs(dD_dt) < threshold_eq[1]
  
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