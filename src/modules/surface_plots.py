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

def surface_plots(model_name='nn', g_plot = 1.76):

  #Set the plot parameters
  scale_surface = 200
  n_sq = 24 * scale_surface
  B_lim = 3
  D_lim = 0.8
  threshold_eq = 1e-4

  # Make a grid of B and D values
  D_edges = np.linspace(0, D_lim, n_sq)
  B_edges = np.linspace(0, B_lim, n_sq)
  D_grid, B_grid = np.meshgrid(D_edges, B_edges)

  # Format the data to feed to the model
  X_pred = np.column_stack((B_grid.flatten(), D_grid.flatten(),
                            np.full(n_sq**2, g_plot)))

  # Load the model
  model = load_model(model_name)

  # Use the model to predict the value of the derivatives in the grid points
  Y_pred = model.predict(X_pred).reshape((n_sq, n_sq, -1))

  # Find stable and unstable equilibrium points
  Y_eq = find_eq_points(Y_pred, threshold_eq, B_grid, D_grid, scale_surface)

  # Plot the surfaces
  plot_surfaces(Y_pred, Y_eq, B_grid, D_grid, scale_surface, model_name)

  # Plot the streamplot
  plot_stream(Y_pred, B_grid, D_grid, model_name)

  # Plot the equilibrium lines
  plot_eq_lines(Y_eq, B_grid, D_grid, model_name)

  # Add effect of threshold_eq to the equilibrium lines plot
  threshold_var = threshold_eq * 10
  Y_eq_var = find_eq_points(Y_pred, threshold_var, B_grid, D_grid, scale_surface)
  plot_eq_lines_var(Y_eq, Y_eq_var, B_grid, D_grid, model_name + '_var')

  # Save the results
  print('Saving surface plot results...')
  df = pd.DataFrame({'B_grid':B_grid.flatten(), 'D_grid':D_grid.flatten(),
                     'dB_dt':Y_pred[:,:,0].flatten(), 'dD_dt':Y_pred[:,:,1].flatten()})
  df.to_csv(paths.outputs / f'surface_plots_{model_name}.csv')
  print('Successfully saved surface plot results.')

  # Add a couple lines to the summary with the system evolution parameters
  surface_summary = "".join(['\n\n*SURFACE PLOTS*',
                             '\nn_sq = {}'.format(n_sq),
                             '\nscale_factor = {}'.format(scale_surface),
                             '\nthreshold_eq = {}'.format(threshold_eq),
                             '\nB_lim = {}'.format(B_lim),
                             '\nD_lim = {}'.format(D_lim),
                             '\ng = {}'.format(g_plot)])
  
  print('Successfully plotted the surface plots.')

  return surface_summary


def load_model(model_name):

  # If the model is a neural network, load it using keras
  if model_name == 'nn':
    model = keras_load_model(paths.models / 'nn_model.h5', compile=False)

  # If the model is a random forest, load it using joblib
  elif model_name == 'rf':
    model = jb.load(paths.models / 'rf_model.joblib')

  # If the model is the minimal model, define it
  elif model_name == 'mm':
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

def find_eq_points(Y_pred, threshold_eq, B_grid, D_grid, scale_surface):

  # Unpack the results
  dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]

  # Get the limits of the data
  B_lim, D_lim = np.max(B_grid), np.max(D_grid)

  # Find the equilibrium points given the threshold weighted by the std
  B_eq = np.abs(dB_dt) < threshold_eq * np.std(dB_dt)
  D_eq = np.abs(dD_dt) < threshold_eq * np.std(dD_dt)
  
  # Compute the gradients
  dB_dt_B, dB_dt_D = np.gradient(dB_dt)
  dD_dt_B, dD_dt_D = np.gradient(dD_dt)

  frac = 1 / (24 * scale_surface)
  
  # Find the stable and unstable equilibrium points
  B_st_eq = B_eq & ((dB_dt_B < 0) | (B_grid < frac * B_lim)) #& (dB_dt_D < 0)
  B_un_eq = B_eq & ~B_st_eq

  D_st_eq = D_eq & ((dD_dt_D < 0) | (D_grid < frac * D_lim)) #& (dD_dt_B < 0 )
  D_un_eq = D_eq & ~D_st_eq

  # Store the results in a dictionary
  Y_eq = {
    'B': {'stable': B_st_eq, 'unstable': B_un_eq},
    'D': {'stable': D_st_eq, 'unstable': D_un_eq}
  }
  
  return Y_eq

def plot_surfaces(Y_pred, Y_eq, B_grid, D_grid, scale_surface, model_name):

  # Unpack the results
  dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]
  B_eq, D_eq = Y_eq['B'], Y_eq['D']

  # Get the limits of the data
  B_lim, D_lim = np.max(B_grid), np.max(D_grid)

  # Plot the surface for dB/dt and dD/dt for both models
  fig, ax = plt.subplots(1,2,figsize=(21,9), subplot_kw={"projection": "3d"})

  # Tweak the font size and resolution
  rcParams['font.size'] = 15
  rcParams['figure.dpi'] = 150

  # Set the style and font
  plt.style.use('default')
  #plt.rcParams['font.family'] = 'Merriweather'

  # Format axis
  min_max_dB = [np.min(dB_dt), np.max(dB_dt)]
  ax[0].set_zticks(min_max_dB,min_max_dB)
  min_max_dD = [np.min(dD_dt), np.max(dD_dt)]
  ax[1].set_zticks(min_max_dD,min_max_dD)
  ax[0].get_proj = lambda: np.dot(Axes3D.get_proj(ax[0]), np.diag([1, 1, 0.3, 1]))
  ax[1].get_proj = lambda: np.dot(Axes3D.get_proj(ax[1]), np.diag([1, 1, 0.3, 1]))
  ax[0].zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
  ax[1].zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
  ax[0].set_zlabel('Biomass net\ngrowth ($kg/m^2/yr$)', labelpad=45, fontsize=20)
  ax[1].set_zlabel('Soil depth\nincrease (m/yr)', labelpad=45, fontsize=20)

  for ax_ in ax:
    ax_.xaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))
    ax_.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax_.tick_params(axis='both', which='major', labelsize=20)
    ax_.tick_params(axis='z', pad=15, labelsize=20)
    ax_.set_xlim(B_lim,0)
    ax_.set_ylim(0,D_lim)
    ax_.set_xlabel('Biomass ($kg/m^2$)', labelpad=25, fontsize=22)
    ax_.set_ylabel('Soil depth ($m$)', labelpad=25, fontsize=22)

  # Create a desaturated version of the colormap
  my_cmap = plt.cm.jet
  desaturation = 0.8
  jet_colors = my_cmap(np.arange(my_cmap.N))
  jet_colors_hsv = mc.rgb_to_hsv(jet_colors[:, :3])
  jet_colors_hsv[:, 1] *= desaturation
  jet_colors_desaturated = mc.hsv_to_rgb(jet_colors_hsv)
  my_cmap_desaturated = mc.ListedColormap(jet_colors_desaturated)

  # Plot the surface for dB/dt
  ax[0].plot_surface(B_grid, D_grid, dB_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black',
            alpha=1, shade=False, rstride=scale_surface, cstride=scale_surface, zorder=1)
  ax[0].set_zlim(np.min(dB_dt),np.max(dB_dt))
  
  # Plot the equilibrium points for dB/dt
  ax[0].plot(B_grid[B_eq['unstable']], D_grid[B_eq['unstable']], dB_dt[B_eq['unstable']],
              color='dimgray', linestyle='', marker='o', markersize=5, zorder=4)
  ax[0].plot(B_grid[B_eq['stable']], D_grid[B_eq['stable']], dB_dt[B_eq['stable']], 
             color='k', linestyle='', marker='o', markersize=5, zorder=5) 
  
  # Plot the surface and eq. lines for dD/dt
  ax[1].plot_surface(B_grid, D_grid, dD_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black',
                    alpha=1, shade=False, rstride=scale_surface, cstride=scale_surface, zorder=1)
  ax[1].set_zlim(np.min(dD_dt),np.max(dD_dt))

  # Plot the equilibrium points for dD/dt
  ax[1].plot(B_grid[D_eq['unstable']], D_grid[D_eq['unstable']], dD_dt[D_eq['unstable']],
             color='dimgray', linestyle='', marker='o', markersize=5, zorder=4)
  ax[1].plot(B_grid[D_eq['stable']], D_grid[D_eq['stable']], dD_dt[D_eq['stable']],
             color='k', linestyle='', marker='o', markersize=5, zorder=5)
  
  plt.tight_layout()
  plt.savefig(paths.figures / f'surface_plot_{model_name}.png')
  #plt.savefig(os.path.join('results','surface_plot_nn.eps'), format='eps')

  return

def plot_stream(Y_pred, B_grid, D_grid, model_name):

  # Unpack the results
  dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]

  # Get the limits of the data
  B_lim, D_lim = np.max(B_grid), np.max(D_grid)

  # Find the feature space velocity and take the logarithm
  velocity = np.sqrt((dB_dt/B_lim)**2 + (dD_dt/D_lim)**2)
  log_vel = np.log10(velocity)

  # Make the streamplot
  fig, ax = plt.subplots(figsize=(16,14), dpi=300)

  stream = plt.streamplot(D_grid, B_grid, dD_dt, dB_dt, color=log_vel, cmap=plt.cm.viridis,
    minlength=0.01, linewidth=3, arrowsize=3)
  cbar = fig.colorbar(stream.lines)
  cbar.set_label('Log reltive rate of change ($s^{-1}$)', size=33, labelpad=17)
  cbar.ax.tick_params(labelsize=35)
  ax.set_ylim(0, B_lim)
  ax.set_xlim(0, D_lim)
  ax.set_ylabel('Biomass ($kg/m^2$)', fontsize=35, labelpad=15)
  ax.set_xlabel('Soil depth ($m$)', fontsize=35, labelpad=15)
  ax.xaxis.set_tick_params(labelsize=35)
  ax.yaxis.set_tick_params(labelsize=35)
  ticks = ax.get_xticks().tolist()
  del ticks[0]
  ax.set_xticks(ticks)
  plt.tight_layout()
  plt.savefig(paths.figures / f'streamplot_{model_name}.png')
  
  return

def plot_eq_lines(Y_eq, B_grid, D_grid, model_name):

  # Unpack the results
  B_eq, D_eq = Y_eq['B'], Y_eq['D']

  # Get the limits of the data
  B_lim, D_lim = np.max(B_grid), np.max(D_grid)

  # Find the equilibrium points of the whole system
  Y_eq_st = B_eq['stable'] & D_eq['stable']
  Y_eq_un = (B_eq['unstable'] & D_eq['unstable']) | \
            (B_eq['stable'] & D_eq['unstable']) | \
            (B_eq['unstable'] & D_eq['stable'])

  # Plot the equilibrium lines
  fig, ax = plt.subplots(figsize=(16,14))
  ax.xaxis.set_major_locator(plt.MaxNLocator(3))
  ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))

  # Plot the equilibrium lines for dB/dt in green
  ax.scatter(D_grid[B_eq['unstable']], B_grid[B_eq['unstable']],
             color='lightgreen', s=10, label='Unstable equilibrium')
  ax.scatter(D_grid[B_eq['stable']], B_grid[B_eq['stable']],
              color='green', s=10, label='Stable equilibrium')
  
  # Plot the equilibrium lines for dD/dt in blue
  ax.scatter(D_grid[D_eq['unstable']], B_grid[D_eq['unstable']],
             color='lightblue', s=10, label='Unstable equilibrium')
  ax.scatter(D_grid[D_eq['stable']], B_grid[D_eq['stable']],
              color='blue', s=10, label='Stable equilibrium')
  
  # Plot the points where Y is in usntable equilibrium in yellow
  ax.scatter(D_grid[Y_eq_un], B_grid[Y_eq_un], color='yellow',
             s=10, label='Unstable equilibrium')
  
  # Plot the points where Y is in stable equilibrium in red
  ax.scatter(D_grid[Y_eq_st], B_grid[Y_eq_st], color='red',
             s=10, label='Stable equilibrium')

  ax.set_ylim(0, B_lim)
  ax.set_xlim(0, D_lim)
  ax.set_ylabel('Biomass ($kg/m^2$)')
  ax.set_xlabel('Soil depth ($m$)')
  ax.legend(loc = 'best', framealpha=1)
  plt.tight_layout()
  plt.savefig(paths.figures / f'eq_lines_{model_name}.png')
  #plt.savefig(os.path.join('results','eq_lines_nn.eps'), format='eps')

  return

def plot_eq_lines_var(Y_eq, Y_eq_var, B_grid, D_grid, model_name):

  # Unpack the results
  B_eq, D_eq = Y_eq['B'], Y_eq['D']
  B_eq_var, D_eq_var = Y_eq_var['B'], Y_eq_var['D']

  # Get the limits of the data
  B_lim, D_lim = np.max(B_grid), np.max(D_grid)

  # Find the equilibrium points of the whole system
  Y_eq_st = B_eq['stable'] & D_eq['stable']
  Y_eq_un = (B_eq['unstable'] & D_eq['unstable']) | \
            (B_eq['stable'] & D_eq['unstable']) | \
            (B_eq['unstable'] & D_eq['stable'])

  # Plot the equilibrium lines
  fig, ax = plt.subplots(figsize=(16,14))
  ax.xaxis.set_major_locator(plt.MaxNLocator(3))
  ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))

  # Plot the equilibrium lines for dB/dt in light green
  ax.scatter(D_grid[B_eq_var['unstable']], B_grid[B_eq_var['unstable']],
        color='lightgreen', s=10, label='Biomass equilibrium var')
  ax.scatter(D_grid[B_eq_var['stable']], B_grid[B_eq_var['stable']],
        color='lightgreen', s=10)

  # Plot the equilibrium lines for dD/dt in light blue
  ax.scatter(D_grid[D_eq_var['unstable']], B_grid[D_eq_var['unstable']],
        color='lightblue', s=10, label='Soil depth equilibrium var')
  ax.scatter(D_grid[D_eq_var['stable']], B_grid[D_eq_var['stable']],
        color='lightblue', s=10)

  # Plot the equilibrium lines for dB/dt in green
  ax.scatter(D_grid[B_eq['unstable']], B_grid[B_eq['unstable']],
             color='green', s=10, label='Biomass equilibrium')
  ax.scatter(D_grid[B_eq['stable']], B_grid[B_eq['stable']],
              color='green', s=10)
  
  # Plot the equilibrium lines for dD/dt in blue
  ax.scatter(D_grid[D_eq['unstable']], B_grid[D_eq['unstable']],
             color='blue', s=10, label='Soil depth equilibrium')
  ax.scatter(D_grid[D_eq['stable']], B_grid[D_eq['stable']],
              color='blue', s=10)
  
  # Plot the points where Y is in usntable equilibrium in yellow
  ax.scatter(D_grid[Y_eq_un], B_grid[Y_eq_un], color='yellow',
             s=10, label='Unstable equilibrium')
  
  # Plot the points where Y is in stable equilibrium in red
  ax.scatter(D_grid[Y_eq_st], B_grid[Y_eq_st], color='red',
             s=10, label='Stable equilibrium')

  ax.set_ylim(0, B_lim)
  ax.set_xlim(0, D_lim)
  ax.set_ylabel('Biomass ($kg/m^2$)')
  ax.set_xlabel('Soil depth ($m$)')
  ax.legend(loc = 'best', framealpha=1)
  plt.tight_layout()
  plt.savefig(paths.figures / f'eq_lines_{model_name}.png')
  #plt.savefig(os.path.join('results','eq_lines_nn.eps'), format='eps')

  return