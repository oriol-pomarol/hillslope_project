import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def data_formatting(X_jumps, y_jumps, X_lin, y_lin):

  # Split the test data
  test_size = 0.2

  # Split the linear data
  X_lin_train, X_lin_test, y_lin_train, y_lin_test = \
    train_test_split(X_lin, y_lin, test_size=test_size,
                     shuffle=True, random_state=10)
  
  # Split the jumps data
  n_sim_test = int(len(X_jumps)*test_size)
  X_jumps_train = np.concatenate(X_jumps[:-n_sim_test])
  y_jumps_train = np.concatenate(y_jumps[:-n_sim_test])
  X_jumps_test = np.concatenate(X_jumps[-n_sim_test:])
  y_jumps_test = np.concatenate(y_jumps[-n_sim_test:])
  
  # Join the linear and jumps data
  X_train_val = np.concatenate((X_lin_train, X_jumps_train))
  y_train_val = np.concatenate((y_lin_train, y_jumps_train))
  X_test = np.concatenate((X_lin_test, X_jumps_test))
  y_test = np.concatenate((y_lin_test, y_jumps_test))

  # Split between training and validation data
  val_size = 0.1
  X_train, X_val, y_train, y_val = \
    train_test_split(X_train_val, y_train_val,
                     test_size=val_size/(1-test_size),
                     shuffle=True, random_state=10)
  
#   # Add the data characteristics to the summary
#   run_summary += "".join(['\n\n***DATA***',
#                           '\ntrain_samples = {}'.format(X_train.shape[0]),
#                           '\ntest_size = {}'.format(test_size),
#                           '\nval_size = {}'.format(val_size)])

  return X_train, X_val, X_test, y_train, y_val, y_test