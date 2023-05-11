import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import matplotlib.ticker as tck
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mc
import os

def surface_plots(model, rforest):

  def eq_lines(contour, gradient, B_lim, D_lim, n_sq):
    # Find for what regions the equilibrium is stable
    grad_stab = gradient < 0

    dashed_lines = []
    solid_lines = [] 
    lines = contour.allsegs[0]
    for line in lines:
      indices = (np.array(line)//np.array([(B_lim+1E-5)/n_sq, (D_lim+1E-6)/n_sq])).astype(int)
      stability = grad_stab[indices[:,0], indices[:,1]]
      current_line = line[0]

      for i in range(len(line)-1):
        if (stability[i] != stability[i+1]):
          midpoint = (line[i] + line[i+1])/2
          current_line = np.vstack([current_line, midpoint])
          if stability[i]:
            solid_lines.append(current_line)
          else:
            dashed_lines.append(current_line)
          current_line = midpoint
        current_line = np.vstack([current_line, line[i+1]])

      if stability[-1]:
            solid_lines.append(current_line)
      else:
        dashed_lines.append(current_line)
    return solid_lines, dashed_lines

  #Set the parameters
  scale_surface = 5
  n_sq = 18 * scale_surface
  B_lim = 3
  D_lim = 0.5
  g_plot = 1.76 #1.76

  # Make a grid of B and D values
  D_edges = np.linspace(0, D_lim, n_sq)
  B_edges = np.linspace(0, B_lim, n_sq)
  D_grid, B_grid = np.meshgrid(D_edges, B_edges)

  # Format the data to feed to the model
  g_grid =  np.ones((n_sq**2)) * g_plot
  X_grid = np.column_stack((B_grid.flatten(), D_grid.flatten(), g_grid))

  # Use RF and NN to predict the value of the derivatives in the grid points
  Z_rf = rforest.predict(X_grid)
  Z_nn = model.predict(X_grid)

  dB_dt_nn = Z_nn[:,0].reshape((n_sq,n_sq))
  dD_dt_nn = Z_nn[:,1].reshape((n_sq,n_sq))

  dB_dt_rf = Z_rf[:,0].reshape((n_sq,n_sq))
  dD_dt_rf = Z_rf[:,1].reshape((n_sq,n_sq))

  # Plot the surface for dB/dt and dD/dt for the RF
  fig, ax = plt.subplots(1,2,figsize=(19,8),subplot_kw={"projection": "3d"})

  rcParams['figure.dpi'] = 150
  rcParams['savefig.transparent'] = True

  ax[0].plot_surface(B_grid, D_grid, dB_dt_rf, cmap='jet', linewidth=0, alpha = 0.5, zorder = 1)
  ax[0].plot_wireframe(B_grid, D_grid, dB_dt_rf, linewidth=0.1, color='k')
  ax[0].contour3D(X=B_grid, Y=D_grid, Z=dB_dt_rf, levels = [0.0], colors='k',
                  linestyles='solid', linewidths=3)
  ax[0].set_xlim(B_lim,0)
  ax[0].set_ylim(0,D_lim)
  ax[0].set_xlabel('Biomass (kg/m2)')
  ax[0].set_ylabel('Soil depth (m)')
  ax[0].set_zlabel('Net growth (kg/m2/yr)')

  ax[1].plot_surface(B_grid, D_grid, dD_dt_rf, cmap='jet', linewidth=0, alpha = 0.5)
  ax[1].plot_wireframe(B_grid, D_grid, dD_dt_rf, linewidth=0.1, color='k')
  ax[1].contour3D(X=B_grid, Y=D_grid, Z=dD_dt_rf, levels = [0.0], colors='k', linestyles='solid', linewidths=3)
  ax[1].set_xlim(B_lim,0)
  ax[1].set_ylim(0,D_lim)
  ax[1].set_xlabel('Biomass (kg/m2)')
  ax[1].set_ylabel('Soil depth (m)')
  ax[1].set_zlabel('Soil depth increase (m/yr)')

  fig.suptitle('Random forest')
  plt.savefig(os.path.join('results','surface_plot_rf.png'))

  # Plot the surface for dB/dt and dD/dt for the NN
  # Change the font sizes
  plt.rcParams['font.size'] = 15
  # plt.rcParams['font.weight'] = 'semibold'

  # plt.rc('font', size=other_size)          # controls default text sizes
  # plt.rc('axes', titlesize=other_size)     # fontsize of the axes title
  # plt.rc('axes', labelsize=label_size)      # fontsize of the x and y labels
  # plt.rc('xtick', labelsize=other_size)    # fontsize of the tick labels
  # plt.rc('ytick', labelsize=other_size)    # fontsize of the tick labels

  fig, ax = plt.subplots(1,2,figsize=(21,9), subplot_kw={"projection": "3d"})

  # Format axis
  min_max_dB = [np.min(dB_dt_nn), np.max(dB_dt_nn)]
  ax[0].set_zticks(min_max_dB,min_max_dB)
  min_max_dD = [np.min(dD_dt_nn), np.max(dD_dt_nn)]
  ax[1].set_zticks(min_max_dD,min_max_dD)
  ax[0].get_proj = lambda: np.dot(Axes3D.get_proj(ax[0]), np.diag([1, 1, 0.3, 1]))
  ax[1].get_proj = lambda: np.dot(Axes3D.get_proj(ax[1]), np.diag([1, 1, 0.3, 1]))
  ax[0].zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
  ax[1].zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
  ax[0].set_zlabel('Biomass net\ngrowth ($kg/m^2/yr$)', labelpad=34)
  ax[1].set_zlabel('Soil depth\nincrease (m/yr)', labelpad=36)

  for ax_ in ax:
    ax_.xaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))
    ax_.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax_.tick_params(axis='z', pad=15)
    ax_.set_xlim(B_lim,0)
    ax_.set_ylim(0,D_lim)
    ax_.set_xlabel('Biomass ($kg/m^2$)', labelpad=20)
    ax_.set_ylabel('Soil depth ($m$)', labelpad=24)

  # Create a desaturated version of the colormap
  my_cmap = plt.cm.jet
  desaturation = 0.8
  jet_colors = my_cmap(np.arange(my_cmap.N))
  jet_colors_hsv = mc.rgb_to_hsv(jet_colors[:, :3])
  jet_colors_hsv[:, 1] *= desaturation
  jet_colors_desaturated = mc.hsv_to_rgb(jet_colors_hsv)
  my_cmap_desaturated = mc.ListedColormap(jet_colors_desaturated)

  # Plot the surfaces
  ax[0].plot_surface(B_grid, D_grid, dB_dt_nn, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black',
                     alpha=1, shade=False, rstride=scale_surface, cstride=scale_surface)
  ax[0].set_zlim(np.min(dB_dt_nn),np.max(dB_dt_nn))
  dB_dt_0 = ax[0].contour3D(X=B_grid, Y=D_grid, Z=dB_dt_nn, levels = [0.0], linewidths=0)

  grad_B, _ = np.gradient(dB_dt_nn)
  solid_lines, dashed_lines = eq_lines(dB_dt_0, grad_B, B_lim, D_lim, n_sq)

  for d_line in dashed_lines:
    ax[0].plot(d_line[:,0], d_line[:,1], zs=0, zdir='z', linestyle = 'dashed',
               linewidth=3, color = 'black', zorder=10)

  for i, s_line in enumerate(solid_lines):
    ax[0].plot(s_line[:,0], s_line[:,1], zs=0, zdir='z', linestyle = 'solid',
               linewidth=3, color = 'black', zorder=11)


  # Plot the surfaces
  ax[1].plot_surface(B_grid, D_grid, dD_dt_nn, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black',
                     alpha=1, shade=False, rstride=scale_surface, cstride=scale_surface)
  ax[1].set_zlim(np.min(dD_dt_nn),np.max(dD_dt_nn))
  dD_dt_0 = ax[1].contour3D(X=B_grid, Y=D_grid, Z=dD_dt_nn, levels = [0.0], linewidths=0)

  _, grad_D = np.gradient(dD_dt_nn)
  solid_lines, dashed_lines = eq_lines(dD_dt_0, grad_D, B_lim, D_lim, n_sq)

  for d_line in dashed_lines:
    ax[1].plot(d_line[:,0], d_line[:,1], zs=0, zdir='z', linestyle = 'dashed',
               linewidth=4, color = 'black', zorder=10)

  for i, s_line in enumerate(solid_lines):
    ax[1].plot(s_line[:,0], s_line[:,1], zs=0, zdir='z', linestyle = 'solid',
               linewidth=4, color = 'black', zorder=11)

  # fig.suptitle('Neural network')
  #plt.tight_layout()
  plt.savefig(os.path.join('results','surface_plot_nn.png'))
  plt.savefig(os.path.join('results','surface_plot_nn.eps'), format='eps')

  # Find the gradients
  grad_B, _ = np.gradient(dB_dt_nn)
  _, grad_D = np.gradient(dD_dt_nn)

  # Change the font sizes
  SMALLER_SIZE = 28
  BIGGER_SIZE = 32

  plt.rc('font', size=SMALLER_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=SMALLER_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=BIGGER_SIZE)      # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels

  # Initialize the figure
  fig, ax = plt.subplots(figsize=(16,14))

  ax.xaxis.set_major_locator(plt.MaxNLocator(3))
  ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))
  # ax.tick_params(axis='both', pad=6)


  # Make the equilibrium line plots
  for contour, gradient, color, var in [[dB_dt_0, grad_B, '#24A793', 'biomass'], [dD_dt_0, grad_D, '#C00A35', 'soil depth']]:
    
    solid_lines, dashed_lines = eq_lines(contour, gradient, B_lim, D_lim, n_sq)

    for i, d_line in enumerate(dashed_lines):
      d_line = np.array(d_line)
      ax.plot(d_line[:,1], d_line[:,0], linestyle = 'dashed', linewidth=5, color = color,
              label = f'Unstable {var} nullcline' if i==0 else "")

    for i, s_line in enumerate(solid_lines):
      s_line = np.array(s_line)
      ax.plot(s_line[:,1], s_line[:,0], linestyle = 'solid', linewidth=5, color = color,
              label = f'Stable {var} nullcline' if i==0 else "")

  ax.set_ylim(0, B_lim)
  ax.set_xlim(0, D_lim)
  ax.set_ylabel('Biomass ($kg/m^2$)')
  ax.set_xlabel('Soil depth ($m$)')
  ax.legend(loc = 'best', framealpha=1)
  plt.tight_layout()
  plt.savefig(os.path.join('results','eq_lines_nn.png'))
  plt.savefig(os.path.join('results','eq_lines_nn.eps'), format='eps')


  # Save the results
  print('Saving surface plot results...')
  df = pd.DataFrame({'B_grid':X_grid[:,0], 'D_grid':X_grid[:,1], 'dB_dt_rf':Z_rf[:,0], 'dD_dt_rf':Z_rf[:,1],
                     'dB_dt_nn':Z_nn[:,0], 'dD_dt_nn':Z_nn[:,1]})
  df.to_csv(os.path.join('results','surface_plots.csv'))
  print('Successfully saved surface plot results.')

  # Add a couple lines to the summary with the system evolution parameters
  surface_summary = "".join(['\n\n***SURFACE PLOTS***',
                             '\nn_sq = {}'.format(n_sq),
                             '\nB_lim = {}'.format(B_lim),
                             '\nD_lim = {}'.format(D_lim),
                             '\ng = {}'.format(g_plot)])

  return surface_summary