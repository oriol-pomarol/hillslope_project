import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def data_formatting(X_jumps, y_jumps, X_lin, y_lin, mode='combined'):

  # Split the test data
  test_size = 0.2

  # Split the g_increase data between testing and the rest
  X_lin_train, X_lin_test, y_lin_train, y_lin_test = \
    train_test_split(X_lin, y_lin, test_size=test_size,
                     shuffle=True, random_state=10)
  
  # Split the jumps data between testing and the rest and concatenate
  n_sim_test = int(len(X_jumps)*test_size)
  X_jumps_train = np.concatenate(X_jumps[:-n_sim_test])
  y_jumps_train = np.concatenate(y_jumps[:-n_sim_test])
  X_jumps_test = np.concatenate(X_jumps[-n_sim_test:])
  y_jumps_test = np.concatenate(y_jumps[-n_sim_test:])

  # Split all the data between training and validation
  val_size = 0.1
  X_lin_train, X_lin_val, y_lin_train, y_lin_val = \
    train_test_split(X_lin_train, y_lin_train,
                      test_size=val_size/(1-test_size),
                      shuffle=True, random_state=10)
  X_jumps_train, X_jumps_val, y_jumps_train, y_jumps_val = \
    train_test_split(X_jumps_train, y_jumps_train,
                      test_size=val_size/(1-test_size),
                      shuffle=True, random_state=10)
  
  if mode == 'combined':
    # Join the linear and jumps data
    X_train = np.concatenate((X_lin_train, X_jumps_train))
    y_train = np.concatenate((y_lin_train, y_jumps_train))
    X_test = np.concatenate((X_lin_test, X_jumps_test))
    y_test = np.concatenate((y_lin_test, y_jumps_test))
    X_val = np.concatenate((X_lin_val, X_jumps_val))
    y_val = np.concatenate((y_lin_val, y_jumps_val))

  elif mode == 'sequential':
    # Store both linear and jumps data in a list
    X_train = [X_jumps_train, X_lin_train]
    y_train = [y_jumps_train, y_lin_train]
    X_test = [X_jumps_test, X_lin_test]
    y_test = [y_jumps_test, y_lin_test]
    X_val = [X_jumps_val, X_lin_val]
    y_val = [y_jumps_val, y_lin_val]

  elif mode == 'jumps':
    # Use only the jumps data
    X_train = X_jumps_train
    y_train = y_jumps_train
    X_test = X_jumps_test
    y_test = y_jumps_test
    X_val = X_jumps_val
    y_val = y_jumps_val

  elif mode == 'linear':
    # Use only the linear data
    X_train = X_lin_train
    y_train = y_lin_train
    X_test = X_lin_test
    y_test = y_lin_test
    X_val = X_lin_val
    y_val = y_lin_val

  else:
    raise ValueError('Invalid mode. Valid modes are: combined, jumps, linear')
  
  # Drop a percentage of the training data for better performance
  drop_size = 0

  if drop_size > 0 and mode != 'sequential':
    drop_mask = np.random.choice([True, False], size = len(X_train), p = [1-drop_size, drop_size])
    X_train = X_train[drop_mask]
    y_train = y_train[drop_mask]  

  elif drop_size > 0 and mode == 'sequential':
    for i in range(len(X_train)):
      drop_mask = np.random.choice([True, False], size = len(X_train[i]), p = [1-drop_size, drop_size])
      X_train[i] = X_train[i][drop_mask]
      y_train[i] = y_train[i][drop_mask]

  # Calculate the final training set size
  final_train_size = X_train.shape[0] if mode != 'sequential' \
    else sum([x.shape[0] for x in X_train])
  print('Dropped {:.1f}% of the training data.'.format(100*drop_size))
  print('Final training set size: {}'.format(final_train_size))
  
  # Add the data characteristics to the summary
  data_summary = "".join(['\n\n***DATA FORMATTING***',
                          '\nmode = {}'.format(mode),
                          '\ntest_size = {}'.format(test_size),
                          '\nval_size = {}'.format(val_size),
                          '\ndrop_size = {}'.format(drop_size),
                          '\nfinal_train_size = {}'.format(final_train_size)])

  return data_summary, X_train, X_val, X_test, y_train, y_val, y_test