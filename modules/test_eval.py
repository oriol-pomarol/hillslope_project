import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import mean_squared_error

def test_eval(nnetwork, rforest, X_test, y_test):

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
  plt.savefig('results/predicted_vs_true_rf.png')
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
  fig.colorbar(h1[3], ax=axs[1])
  fig.suptitle('Neural network')
  fig.patch.set_alpha(1)
  plt.savefig('results/predicted_vs_true_nn.png')

  # Save the results
  np.savez('results/test_evaluation.npz', y_test=y_test, y_pred_for=y_pred_for, y_pred_nn=y_pred_nn)
  print('Successfully completed NN test set evaluation.')

  return rf_test_summary, nn_test_summary