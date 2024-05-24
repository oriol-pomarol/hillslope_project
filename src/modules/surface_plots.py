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

def surface_plots(name='nn', g_plot = 1.76):

  #Set the plot parameters
  scale_surface = 20
  n_sq = 18 * scale_surface
  B_lim = 3
  D_lim = 0.6
  threshold_eq = 1e-3

  # Make a grid of B and D values
  D_edges = np.linspace(0, D_lim, n_sq)
  B_edges = np.linspace(0, B_lim, n_sq)
  D_grid, B_grid = np.meshgrid(D_edges, B_edges)

  # Format the data to feed to the model
  X_pred = np.column_stack((B_grid.flatten(), D_grid.flatten(),
                            np.full(n_sq**2, g_plot)))

  # Load the model
  model = load_model(name)

  # Use the model to predict the value of the derivatives in the grid points
  Y_pred = model.predict(X_pred).reshape((n_sq, n_sq, -1))

  # Find stable and unstable equilibrium points
  Y_eq = find_eq_points(Y_pred, threshold_eq)

  # Print the stds
  print('dB/dt std:', np.std(Y_pred[:,:,0]))
  print('dD/dt std:', np.std(Y_pred[:,:,1]))

  # Plot the surfaces
  plot_surfaces(Y_pred, Y_eq, B_grid, D_grid, scale_surface, name)

  #plot_eq_lines()

  #plot_stream()

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

def find_eq_points(Y_pred, threshold_eq):

  # Find the equilibrium points given the threshold weighted by the std
  B_eq = np.abs(Y_pred[:,:,0]) < threshold_eq * np.std(Y_pred[:,:,0])
  D_eq = np.abs(Y_pred[:,:,1]) < threshold_eq * np.std(Y_pred[:,:,1])
  
  # Join them in an array of the same shape as B_eq and D_eq but with an extra dimension
  Y_eq = np.stack((B_eq, D_eq), axis=-1)
  
  return Y_eq

def plot_surfaces(Y_pred, Y_eq, B_grid, D_grid, scale_surface, name):

  # Unpack the results
  dB_dt, dD_dt = Y_pred[:,:,0], Y_pred[:,:,1]
  dB_dt_eq, dD_dt_eq = Y_eq[:,:,0], Y_eq[:,:,1]

  # Get the limits of the data
  B_lim, D_lim = np.max(B_grid), np.max(D_grid)

  # Plot the surface for dB/dt and dD/dt for both models
  fig, ax = plt.subplots(1,2,figsize=(21,9), subplot_kw={"projection": "3d"})

  # Tweak the font size and resolution
  rcParams['font.size'] = 15
  rcParams['figure.dpi'] = 150

  # Set the style and font
  plt.style.use('default')
  plt.rcParams['font.family'] = 'Merriweather'

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
  ax[0].plot(B_grid[dB_dt_eq], D_grid[dB_dt_eq], dB_dt[dB_dt_eq], color='k', linestyle='', marker='o', markersize=5, zorder=4)  
  
  # Plot the surface and eq. lines for dD/dt
  ax[1].plot_surface(B_grid, D_grid, dD_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black',
                    alpha=1, shade=False, rstride=scale_surface, cstride=scale_surface, zorder=1)
  ax[1].set_zlim(np.min(dD_dt),np.max(dD_dt))

  # Plot the equilibrium points for dD/dt
  ax[1].plot(B_grid[dD_dt_eq], D_grid[dD_dt_eq], np.zeros_like(B_grid[dD_dt_eq]), 'ko', linestyle='', markersize=5, zorder=4)
  
  plt.tight_layout()
  plt.savefig(paths.figures / f'surface_plot_{name}.png')
  #plt.savefig(os.path.join('results','surface_plot_nn.eps'), format='eps')

  return

  # # Plot the equilibrium lines
  # fig, ax = plt.subplots(figsize=(16,14))
  # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
  # ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))

  # for solid_lines, dashed_lines, color, var in [[st_eq_B, un_eq_B, '#24A793', 'biomass'], [st_eq_D, un_eq_D, '#C00A35', 'soil depth']]:
    
  #   for i, d_line in enumerate(dashed_lines):
  #     d_line = np.array(d_line)
  #     ax.plot(d_line[:,1], d_line[:,0], linestyle = 'dashed', linewidth=5, color = color,
  #             label = f'Unstable {var} nullcline' if i==0 else "")

  #   for i, s_line in enumerate(solid_lines):
  #     s_line = np.array(s_line)
  #     ax.plot(s_line[:,1], s_line[:,0], linestyle = 'solid', linewidth=5, color = color,
  #             label = f'Stable {var} nullcline' if i==0 else "")

  # ax.set_ylim(0, B_lim)
  # ax.set_xlim(0, D_lim)
  # ax.set_ylabel('Biomass ($kg/m^2$)')
  # ax.set_xlabel('Soil depth ($m$)')
  # ax.legend(loc = 'best', framealpha=1)
  # plt.tight_layout()
  # plt.savefig(paths.figures / f'eq_lines_{name}.png')
  # #plt.savefig(os.path.join('results','eq_lines_nn.eps'), format='eps')

  # # Find the feature space velocity and take the logarithm
  # velocity = np.sqrt((dB_dt/B_lim)**2 + (dD_dt/D_lim)**2)
  # log_vel = np.log10(velocity)

  # # Make the streamplot
  # fig, ax = plt.subplots(figsize=(16,14), dpi=300)

  # stream = plt.streamplot(D_grid, B_grid, dD_dt, dB_dt, color=log_vel, cmap=plt.cm.viridis,
  #   minlength=0.01, linewidth=3, arrowsize=3)
  # cbar = fig.colorbar(stream.lines)
  # cbar.set_label('Log reltive rate of change ($s^{-1}$)', size=33, labelpad=17)
  # cbar.ax.tick_params(labelsize=35)
  # ax.set_ylim(0, B_lim)
  # ax.set_xlim(0, D_lim)
  # ax.set_ylabel('Biomass ($kg/m^2$)', fontsize=35, labelpad=15)
  # ax.set_xlabel('Soil depth ($m$)', fontsize=35, labelpad=15)
  # ax.xaxis.set_tick_params(labelsize=35)
  # ax.yaxis.set_tick_params(labelsize=35)
  # ticks = ax.get_xticks().tolist()
  # del ticks[0]
  # ax.set_xticks(ticks)
  # plt.tight_layout()
  # plt.savefig(paths.figures / f'streamplot_{name}.png')
  
  # # Save the results
  # print('Saving surface plot results...')
  # df = pd.DataFrame({'B_grid':X_grid[:,0], 'D_grid':X_grid[:,1],
  #                    'dB_dt':Z[:,0], 'dD_dt':Z[:,1]})
  # df.to_csv(paths.outputs / f'surface_plots_{name}.csv')
  # print('Successfully saved surface plot results.')