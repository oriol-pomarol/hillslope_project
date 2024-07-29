import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from config import paths

def colormesh_plots():

  # Load the training and validation data from csv files
  X_train = np.loadtxt(paths.processed_data / 'X_train.csv',
                       delimiter=",", skiprows=1)
  y_train = np.loadtxt(paths.processed_data / 'y_train.csv',
                       delimiter=",", skiprows=1)
  
  # Set the parameters
  n_bins = 24
  B_max = 3
  D_max = 0.4
  g_range = (0,3)

  # Initialize the variables for observations, mean and standard deviations 
  obs = np.zeros((n_bins, n_bins), dtype = np.int32)
  mean = np.zeros((n_bins, n_bins, 2))
  std = np.zeros((n_bins, n_bins, 2))

  # Find the variables within the corresponding range for each matrix element
  for i in range(n_bins):
    for j in range(n_bins):
      filter_B = (X_train[:,0] >= i*B_max/n_bins) & (X_train[:,0] < (i+1)*B_max/n_bins)
      filter_D = (X_train[:,1] >= j*D_max/n_bins) & (X_train[:,1] < (j+1)*D_max/n_bins)
      filter_g = (X_train[:,2] >= g_range[0]) & (X_train[:,2] < g_range[1])

      subset = y_train[filter_B & filter_D & filter_g]
      obs[i,j] = subset.shape[0]

      if subset.shape[0] != 0:
        mean[i,j] = np.mean(subset, axis = 0)
        std[i,j] = np.std(subset, axis = 0)
      else:
        mean[i,j] = np.nan
        std[i,j] = np.nan

  n_obs = np.sum(obs)
  print(f"Number of observations: {n_obs} "+
        f"({n_obs*100/X_train.shape[0]:.2f}% of training data).")

  B_edges = np.linspace(0, B_max, n_bins+1)
  D_edges = np.linspace(0, D_max, n_bins+1)

  # Plot the number of observations per combination of B and D
  fig, ax = plt.subplots(figsize = (11,7))
  mesh = ax.pcolormesh(D_edges, B_edges, obs, norm=LogNorm())
  ax.invert_xaxis()
  ax.set_xlabel("Soil depth (D)")
  ax.set_ylabel("Biomass (B)")
  ax.text(0.58, 0.1, f"Total number of observations: {n_obs} "+
                     f"({n_obs*100/X_train.shape[0]:.2f}% of training data).")

  fig.colorbar(mesh, label = "Number of observations")
  plt.savefig(paths.figures / 'colormesh_plot_obs.png')

  # Plot the mean rate of change per combination of B and D
  fig, axs = plt.subplots(1, 2, figsize = (20,7))

  mesh_0 = axs[0].pcolormesh(D_edges, B_edges, mean[:,:,0])
  axs[0].invert_xaxis()
  axs[0].set_xlabel("Soil depth (D)")
  axs[0].set_ylabel("Biomass (B)")
  fig.colorbar(mesh_0, label = "Mean dB/dt", ax=axs[0])

  mesh_1 = axs[1].pcolormesh(D_edges, B_edges, mean[:,:,1])
  axs[1].invert_xaxis()
  axs[1].set_xlabel("Soil depth (D)")
  axs[1].set_ylabel("Biomass (B)")
  fig.colorbar(mesh_1, label = "Mean dD/dt", ax=axs[1])

  plt.savefig(paths.figures / 'colormesh_plot_mean.png')

  # Plot the std of the rate of change per combination of B and D
  fig, axs = plt.subplots(1, 2, figsize = (20,7))

  mesh_0 = axs[0].pcolormesh(D_edges, B_edges, std[:,:,0])
  axs[0].invert_xaxis()
  axs[0].set_xlabel("Soil depth (D)")
  axs[0].set_ylabel("Biomass (B)")
  fig.colorbar(mesh_0, label = "Std. dB/dt", ax=axs[0])

  mesh_1 = axs[1].pcolormesh(D_edges, B_edges, std[:,:,1])
  axs[1].invert_xaxis()
  axs[1].set_xlabel("Soil depth (D)")
  axs[1].set_ylabel("Biomass (B)")
  fig.colorbar(mesh_1, label = "Std. dD/dt", ax=axs[1])

  plt.text(2, 0, f"Number of observations: {n_obs} "+
                  f"({n_obs*100/X_train.shape[0]:.2f}% of training data).")
  plt.savefig(paths.figures / 'colormesh_plot_std.png')

  # Add a couple lines to the summary with the system evolution parameters
  colormesh_summary = "".join(['\n\n*COLORMESH PLOTS*',
                               '\nn_bins = {}'.format(n_bins),
                               '\nB_max = {}'.format(B_max),
                               '\nD_max = {}'.format(D_max),
                               '\ng_range = ({}, {})'.format(g_range[0],g_range[1])])

  return colormesh_summary