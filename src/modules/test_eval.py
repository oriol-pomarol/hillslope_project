import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from config import paths
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def test_eval(nnetwork, rforest):

  # Load the test data
  X_test = np.load(paths.processed_data / 'X_test.npy')
  y_test = np.load(paths.processed_data / 'y_test.npy')

  print('Starting RF test set evaluation...')
  y_pred_for = rforest.predict(X_test)
  mse_for = mean_squared_error(y_test, y_pred_for)
  print('MSE test set (RF) = {:.5g}'.format(mse_for))

  rf_test_summary = '\nmse_test= {}'.format(mse_for)

  # Define some variables for the plots
  min_B_test = min(np.min(y_test[:,0]), np.min(y_pred_for[:,0]))
  max_B_test = max(np.max(y_test[:,0]), np.max(y_pred_for[:,0]))
  B_binwidth = (max_B_test - min_B_test)/50

  min_D_test = min(np.min(y_test[:,1]), np.min(y_pred_for[:,1]))
  max_D_test = max(np.max(y_test[:,1]), np.max(y_pred_for[:,1]))
  D_binwidth = (max_D_test - min_D_test)/50

  # Plot the predicted vs true values for dB/dt and dD/dt of RF
  fig, axs = plt.subplots(1, 2, figsize = (12,6))

  axs[0].plot(y_test[:,0], y_test[:,0], '-r')
  h0 = axs[0].hist2d(y_test[:,0], y_pred_for[:,0], 
                     bins = np.arange(min_B_test, max_B_test + B_binwidth, B_binwidth), 
                     norm = colors.LogNorm())
  axs[0].set_xlabel('True dB/dt values')
  axs[0].set_ylabel('Predicted dB/dt values')
  axs[0].autoscale()

  axs[1].plot(y_test[:,1], y_test[:,1], '-r')
  h1 = axs[1].hist2d(y_test[:,1], y_pred_for[:,1], 
                     bins = np.arange(min_D_test, max_D_test + D_binwidth, D_binwidth),
                     norm = colors.LogNorm())
  axs[1].set_xlabel('True dD/dt values')
  axs[1].set_ylabel('Predicted dD/dt values')
  axs[1].autoscale()

  fig.colorbar(h0[3], ax=axs[0])
  fig.colorbar(h1[3], ax=axs[1])
  fig.suptitle('Random forest')
  fig.patch.set_alpha(1)

  # Calculate R² value for RF for each plot and add it to the plot
  r2_for_0 = r2_score(y_test[:,0], y_pred_for[:,0])
  r2_for_1 = r2_score(y_test[:,1], y_pred_for[:,1])
  axs[0].text(0.05, 0.95, f'R² = {r2_for_0:.2f}', transform=axs[0].transAxes, verticalalignment='top')
  axs[1].text(0.05, 0.95, f'R² = {r2_for_1:.2f}', transform=axs[1].transAxes, verticalalignment='top')
  plt.savefig(paths.figures / 'predicted_vs_true_rf.png')

  
  print('Successfully completed RF test set evaluation.')


  # Predict the values from the test set and evaluate the MSE
  print('Starting NN test set evaluation...')
  y_pred_nn = nnetwork.predict(X_test, verbose = False)
  mse_nn = mean_squared_error(y_test, y_pred_nn)
  print('MSE test set (NN) = {:.5g}'.format(mse_nn))

  nn_test_summary = '\nmse_test= {}'.format(mse_nn)


  # Determine which should be minimum, maximum width of the bins
  min_B_test = min(np.min(y_test[:,0]), np.min(y_pred_nn[:,0]))
  max_B_test = max(np.max(y_test[:,0]), np.max(y_pred_nn[:,0]))
  B_binwidth = (max_B_test - min_B_test)/50

  min_D_test = min(np.min(y_test[:,1]), np.min(y_pred_nn[:,1]))
  max_D_test = max(np.max(y_test[:,1]), np.max(y_pred_nn[:,1]))
  D_binwidth = (max_D_test - min_D_test)/50

  # Plot the predicted vs true values for dB/dt and dD/dt of NN
  fig, axs = plt.subplots(1, 2, figsize = (12,6))

  axs[0].plot(y_test[:,0], y_test[:,0], '-r')
  h0 = axs[0].hist2d(y_test[:,0], y_pred_nn[:,0], 
                     bins = np.arange(min_B_test, max_B_test + B_binwidth, B_binwidth), 
                     norm = colors.LogNorm())
  axs[0].set_xlabel('True dB/dt values')
  axs[0].set_ylabel('Predicted dB/dt values')
  axs[0].autoscale()

  axs[1].plot(y_test[:,1], y_test[:,1], '-r')
  h1 = axs[1].hist2d(y_test[:,1], y_pred_nn[:,1], 
                     bins = np.arange(min_D_test, max_D_test + D_binwidth, D_binwidth),
                     norm = colors.LogNorm())
  axs[1].set_xlabel('True dD/dt values')
  axs[1].set_ylabel('Predicted dD/dt values')
  axs[1].autoscale()

  fig.colorbar(h0[3], ax=axs[0])
  print(min(y_test[:,1]), max(y_test[:,1]), min(y_pred_nn[:,1]), max(y_pred_nn[:,1]))
  fig.colorbar(h1[3], ax=axs[1])
  fig.suptitle('Neural network')
  fig.patch.set_alpha(1)

  # Calculate R² value for NN for each plot and add it to the plot
  r2_nn_0 = r2_score(y_test[:,0], y_pred_nn[:,0])
  r2_nn_1 = r2_score(y_test[:,1], y_pred_nn[:,1])
  axs[0].text(0.05, 0.95, f'R² = {r2_nn_0:.2f}', transform=axs[0].transAxes, verticalalignment='top')
  axs[1].text(0.05, 0.95, f'R² = {r2_nn_1:.2f}', transform=axs[1].transAxes, verticalalignment='top')
  plt.savefig(paths.figures / 'predicted_vs_true_nn.png')

  # Save the results
  np.savez(paths.outputs / 'test_evaluation.npz', y_test=y_test, y_pred_for=y_pred_for, y_pred_nn=y_pred_nn)
  print('Successfully completed NN test set evaluation.')

  return rf_test_summary, nn_test_summary