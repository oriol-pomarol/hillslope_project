import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import joblib as jb
from keras.models import load_model
from config import paths
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def model_evaluation(mode='all'):

  # Load the models
  nnetwork = load_model(paths.models / 'nn_model.h5', compile=False)
  rforest = jb.load(paths.models / 'rf_model.joblib')

  # Initialize the summaries
  rf_eval_summary = ''
  nn_eval_summary = ''

  if mode == 'train' or mode == 'all':

    # Load the training data from csv files
    X_train = np.loadtxt(paths.processed_data / 'X_train.csv',
                         delimiter=",", skiprows=1)
    y_train = np.loadtxt(paths.processed_data / 'y_train.csv',
                         delimiter=",", skiprows=1)
    
    # Evaluate the training set
    print('Starting training set evaluation...')
    rf_train_summary, nn_train_summary = \
      evaluate_set(X_train, y_train, rforest, nnetwork, set_name='train')
    rf_eval_summary += rf_train_summary
    nn_eval_summary += nn_train_summary
    print('Successfully completed training set evaluation.')

  if mode == 'test' or mode == 'all':

    # Load the data from csv files
    X_test = np.loadtxt(paths.processed_data / 'X_test.csv',
                        delimiter=",", skiprows=1)
    y_test = np.loadtxt(paths.processed_data / 'y_test.csv',
                        delimiter=",", skiprows=1)
    
    # Evaluate the test set
    print('Starting test set evaluation...')
    rf_test_summary, nn_test_summary = \
      evaluate_set(X_test, y_test, rforest, nnetwork, set_name='test')
    rf_eval_summary += rf_test_summary
    nn_eval_summary += nn_test_summary
    print('Successfully completed test set evaluation.')

  return rf_eval_summary, nn_eval_summary

##############################################################################

def evaluate_set(X, y, rforest, nnetwork, set_name='train'):

  # Evaluate the MSE for the random forest
  y_pred_for = rforest.predict(X)
  mse_for = mean_squared_error(y, y_pred_for)
  print('MSE {} set (RF) = {:.5g}'.format(set_name, mse_for))

  rf_summary = '\nmse_{}= {}'.format(set_name, mse_for)

  plot_true_vs_pred(y, y_pred_for, set_name, 'rf')

  # Predict the values from the test set and evaluate the MSE
  y_pred_nn = nnetwork.predict(X, verbose = False)
  mse_nn = mean_squared_error(y, y_pred_nn)
  print('MSE test set (NN) = {:.5g}'.format(mse_nn))

  nn_summary = '\nmse_{}= {}'.format(set_name, mse_nn)

  plot_true_vs_pred(y, y_pred_nn, set_name, 'nn')

  # Save the results
  np.savez(paths.outputs / 'test_evaluation.npz', y=y, y_pred_for=y_pred_for, y_pred_nn=y_pred_nn)
  print('Successfully completed NN test set evaluation.')

  return rf_summary, nn_summary

##############################################################################

def plot_true_vs_pred(y, y_pred, set_name, model_name):

  # Plot sizes
  fontsize_labels = 18
  fontsize_ticks = 18

  # Define some variables for the plots
  min_B_test = min(np.min(y[:,0]), np.min(y_pred[:,0]))
  max_B_test = max(np.max(y[:,0]), np.max(y_pred[:,0]))
  B_binwidth = (max_B_test - min_B_test)/50

  min_D_test = min(np.min(y[:,1]), np.min(y_pred[:,1]))
  max_D_test = max(np.max(y[:,1]), np.max(y_pred[:,1]))
  D_binwidth = (max_D_test - min_D_test)/50

  # Define a colormap and a normalization instance
  cmap = plt.cm.viridis
  norm = colors.LogNorm(vmin=1, vmax=len(y))

  # Plot the predicted vs true values for dB/dt and dD/dt of RF
  plt.style.use('seaborn-v0_8')
  fig, axs = plt.subplots(1, 2, figsize = (16,7))

  axs[0].plot(y[:,0], y[:,0], '-k')
  h0 = axs[0].hist2d(y[:,0], y_pred[:,0], 
                     bins = np.arange(min_B_test, max_B_test + B_binwidth, B_binwidth), 
                     cmap = cmap, norm = norm)
  axs[0].set_xlabel('Modelled $\Delta B/\Delta t$ ($kg/m^2/yr$)', fontsize = fontsize_labels)
  axs[0].set_ylabel('Predicted $\Delta B/\Delta t$ ($kg/m^2/yr$)', fontsize = fontsize_labels)
  axs[0].autoscale()

  axs[1].plot(y[:,1], y[:,1], '-k')
  h1 = axs[1].hist2d(y[:,1], y_pred[:,1], 
                     bins = np.arange(min_D_test, max_D_test + D_binwidth, D_binwidth),
                     norm = colors.LogNorm())
  axs[1].set_xlabel('Modelled $\Delta B/\Delta t$ ($kg/m^2/yr$)', fontsize = fontsize_labels)
  axs[1].set_ylabel('Predicted $\Delta B/\Delta t$ ($kg/m^2/yr$)', fontsize = fontsize_labels)
  axs[1].autoscale()

  # Calculate R² value for RF for each plot and add it to the plot
  r2_for_0 = r2_score(y[:,0], y_pred[:,0])
  r2_for_1 = r2_score(y[:,1], y_pred[:,1])
  axs[0].text(0.05, 0.95, f'R² = {r2_for_0:.2e}', transform=axs[0].transAxes,
              verticalalignment='top', fontsize=fontsize_labels)
  axs[1].text(0.05, 0.95, f'R² = {r2_for_1:.2e}', transform=axs[1].transAxes,
              verticalalignment='top', fontsize=fontsize_labels)
  
  # Use MaxNLocator for nice tick values
  n_ticks = 4
  axs[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=n_ticks+1))
  axs[0].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=n_ticks+1))
  axs[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=n_ticks))
  axs[1].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=n_ticks))

  # Adjust each subplot to allow for space for the axis labels
  plt.subplots_adjust(wspace=0.25)

  # Create a single colorbar for the entire figure
  cbar_ax = fig.add_axes([0.91, 0.12, 0.03, 0.75])
  cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
  cbar.set_label('Number of data points', fontsize=18)
  cbar.ax.tick_params(labelsize=18)
  cbar.ax.tick_params(axis='both', which='major', length=3, width=1)

  # Set font size for tick labels
  for ax in axs:
      ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

  plt.savefig(paths.figures / f'{set_name}_pred_vs_true_{model_name}.png')

  return