import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def train_eval(rforest, nnetwork, train_val_data):

  # Unpack the data
  X_train, X_val, y_train, y_val = train_val_data
  # Check if the input is a list of datasets
  if isinstance(X_train, list):
    rf_sum = ''
    nn_sum = ''
    for i in range(len(X_train)):
      rf_sum_add, nn_sum_add = train_eval_single(rforest, nnetwork,
                                                 X_train[i], y_train[i],
                                                 X_val[i], y_val[i], i)
      rf_sum += rf_sum_add
      nn_sum += nn_sum_add
    return rf_sum, nn_sum
  else:
    return train_eval_single(rforest, nnetwork, X_train, y_train, X_val, y_val)

def train_eval_single(rforest, nnetwork, X_train, y_train, X_val, y_val, iter=0):
  # Evalute RF model on validation and training data
  print('Starting RF train set evaluation...')
  y_pred_train_for = rforest.predict(X_train)
  mse_train_for = mean_squared_error(y_train, y_pred_train_for)
  y_pred_val_for = rforest.predict(X_val)
  mse_val_for = mean_squared_error(y_val, y_pred_val_for)
  print('MSE train set (RF) = {:.5g}'.format(mse_train_for))
  print('MSE validation set (RF) = {:.5g}'.format(mse_val_for))
  print('Successfully completed RF train set evaluation.')
  rf_train_summary = "".join(['\nmse_train_{} = {}'.format(iter, mse_train_for),
                              '\nmse_val_{} = {}'.format(iter,mse_val_for)])

  # Define some variables for the plots
  min_B_test = min(np.min(y_train[:,0]), np.min(y_pred_train_for[:,0]))
  max_B_test = max(np.max(y_train[:,0]), np.max(y_pred_train_for[:,0]))
  B_binwidth = (max_B_test - min_B_test)/50

  min_D_test = min(np.min(y_train[:,1]), np.min(y_pred_train_for[:,1]))
  max_D_test = max(np.max(y_train[:,1]), np.max(y_pred_train_for[:,1]))
  D_binwidth = (max_D_test - min_D_test)/50

  # Plot the predicted vs true values for dB/dt and dD/dt of RF
  fig, axs = plt.subplots(1, 2, figsize = (12,6))

  axs[0].plot(y_train[:,0], y_train[:,0], '-r')
  h0 = axs[0].hist2d(y_train[:,0], y_pred_train_for[:,0], 
                     bins = np.arange(min_B_test, max_B_test + B_binwidth, B_binwidth), 
                     norm = colors.LogNorm())
  axs[0].set_xlabel('True dB/dt values')
  axs[0].set_ylabel('Predicted dB/dt values')
  axs[0].autoscale()

  axs[1].plot(y_train[:,1], y_train[:,1], '-r')
  h1 = axs[1].hist2d(y_train[:,1], y_pred_train_for[:,1], 
                     bins = np.arange(min_D_test, max_D_test + D_binwidth, D_binwidth),
                     norm = colors.LogNorm())
  axs[1].set_xlabel('True dD/dt values')
  axs[1].set_ylabel('Predicted dD/dt values')
  axs[1].autoscale()

  fig.colorbar(h0[3], ax=axs[0])
  fig.colorbar(h1[3], ax=axs[1])
  fig.suptitle('Random forest' + '' if iter is None else f' (dataset {iter})')
  fig.patch.set_alpha(1)

  # Add R² value to the plot for RF
  r2_train_for_0 = r2_score(y_train[:,0], y_pred_train_for[:,0])
  r2_train_for_1 = r2_score(y_train[:,1], y_pred_train_for[:,1])
  axs[0].text(0.05, 0.95, f'R² = {r2_train_for_0:.2f}', transform=axs[0].transAxes, verticalalignment='top')
  axs[1].text(0.05, 0.95, f'R² = {r2_train_for_1:.2f}', transform=axs[1].transAxes, verticalalignment='top')

  if iter is None:
    plt.savefig('results/train_predicted_vs_true_rf.png')
  else:
    plt.savefig(f'results/train_predicted_vs_true_rf_{iter}.png')

  # Evalute NN model on validation and training data
  print('Starting NN train set evaluation...')
  y_pred_train_nn = nnetwork.predict(X_train)
  mse_train_nn = mean_squared_error(y_train, y_pred_train_nn)
  y_pred_val_nn = nnetwork.predict(X_val)
  mse_val_nn = mean_squared_error(y_val, y_pred_val_nn)
  print('MSE train set (NN) = {:.5g}'.format(mse_train_nn))
  print('MSE validation set (NN) = {:.5g}'.format(mse_val_nn))
  print('Successfully completed NN train set evaluation.')
  nn_train_summary = "".join(['\nmse_train_{} = {}'.format(iter, mse_train_nn),
                              '\nmse_val_{} = {}'.format(iter, mse_val_nn)])
  # Determine which should be minimum, maximum width of the bins
  min_B_test = min(np.min(y_train[:,0]), np.min(y_pred_train_nn[:,0]))
  max_B_test = max(np.max(y_train[:,0]), np.max(y_pred_train_nn[:,0]))
  B_binwidth = (max_B_test - min_B_test)/50

  min_D_test = min(np.min(y_train[:,1]), np.min(y_pred_train_nn[:,1]))
  max_D_test = max(np.max(y_train[:,1]), np.max(y_pred_train_nn[:,1]))
  D_binwidth = (max_D_test - min_D_test)/50

  # Plot the predicted vs true values for dB/dt and dD/dt of NN
  fig, axs = plt.subplots(1, 2, figsize = (12,6))

  axs[0].plot(y_train[:,0], y_train[:,0], '-r')
  h0 = axs[0].hist2d(y_train[:,0], y_pred_train_nn[:,0], 
                     bins = np.arange(min_B_test, max_B_test + B_binwidth, B_binwidth), 
                     norm = colors.LogNorm())
  axs[0].set_xlabel('True dB/dt values')
  axs[0].set_ylabel('Predicted dB/dt values')
  axs[0].autoscale()

  axs[1].plot(y_train[:,1], y_train[:,1], '-r')
  h1 = axs[1].hist2d(y_train[:,1], y_pred_train_nn[:,1], 
                     bins = np.arange(min_D_test, max_D_test + D_binwidth, D_binwidth),
                     norm = colors.LogNorm())
  axs[1].set_xlabel('True dD/dt values')
  axs[1].set_ylabel('Predicted dD/dt values')
  axs[1].autoscale()

  fig.colorbar(h0[3], ax=axs[0])
  fig.colorbar(h1[3], ax=axs[1])
  fig.suptitle('Neural network' + '' if iter is None else f' (dataset {iter})')
  fig.patch.set_alpha(1)

  # Add R² value to the plot for NN
  r2_train_nn_0 = r2_score(y_train[:,0], y_pred_train_nn[:,0])
  r2_train_nn_1 = r2_score(y_train[:,1], y_pred_train_nn[:,1])
  axs[0].text(0.05, 0.95, f'R² = {r2_train_nn_0:.2f}', transform=axs[0].transAxes, verticalalignment='top')
  axs[1].text(0.05, 0.95, f'R² = {r2_train_nn_1:.2f}', transform=axs[1].transAxes, verticalalignment='top')

  if iter is None:
    plt.savefig('results/train_predicted_vs_true_nn.png')
  else:
    plt.savefig(f'results/train_predicted_vs_true_nn_{iter}.png')
  
  return rf_train_summary, nn_train_summary
