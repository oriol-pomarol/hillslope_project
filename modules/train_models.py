import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
import os
import joblib as jb

def train_models(X_train, X_val, y_train, y_val, mode='all'):

  # Define some internal parameters
  n_bins = 6
  n_epochs = 20

  if (mode=='rf' or mode=='all'):
    # Start the random forest model training
    print('Starting Random Forest training...')
    rforest = RandomForestRegressor(n_estimators = 100, max_features = 'sqrt', 
      max_samples = 0.4, min_samples_leaf = 2, min_samples_split = 20)
    rforest.fit(X_train, y_train)
    jb.dump(rforest, 'data/rf_model.joblib') 
    print('Successfully completed Random Forest training.')

  if (mode=='nn' or mode=='all'):
    # Set all layers to use float64
    tf.keras.backend.set_floatx('float64')

    # Obtain the standard deviations of the training data
    dB_dt_std = np.std(y_train[:,0])
    dD_dt_std = np.std(y_train[:,1])

    #define a loss function
    def custom_mse(y_true, y_pred):
      loss = y_pred - y_true
      loss = loss / [dB_dt_std, dD_dt_std]
      loss = K.square(loss)
      loss = K.sum(loss, axis=1) 
      return loss
    
    # Perform binning
    bin_edges = np.array([np.linspace(0, 3, n_bins + 1), 
                          np.linspace(0, 0.6, n_bins + 1), 
                          np.linspace(0, 3, n_bins + 1)])

    hp = {'units':[9, 27, 81, 162, 324, 648, 1296], 'act_fun':'relu', 
          'learning_rate':1E-5, 'batch_size':64}

    # Define the model
    nnetwork = keras.Sequential()
    for n_units in hp['units']:
      nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=hp['act_fun']))
    nnetwork.add(keras.layers.Dense(2, activation='linear'))

    # Compile the model
    print('Starting Neural Network training...')
    nnetwork.compile(optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate']), loss=custom_mse)
    
    # Define lists to store train and validation metrics for each bin
    train_metrics_per_bin = []
    val_metrics_per_bin = []

    for epoch in range(n_epochs):
      history = nnetwork.fit(X_train, y_train, epochs = 1, validation_data = (X_val, y_val), 
                           batch_size = hp['batch_size'])
      nnetwork.save(os.path.join('data', 'checkpoints', f'nn_epoch_{epoch}.h5'))

      # Calculate metrics for each bin
      train_metrics = []
      val_metrics = []

      for i in range(n_bins):
        for j in range(n_bins):
          for k in range(n_bins):
            # Filter data based on current bin for each feature
            bin_masks_train = []
            bin_masks_val = []
            bin_mask_train = np.logical_and.reduce([X_train[:,0] >= bin_edges[0][i], 
                                                    X_train[:,0] < bin_edges[0][i + 1],
                                                    X_train[:,1] >= bin_edges[1][j],
                                                    X_train[:,1] < bin_edges[1][j + 1],
                                                    X_train[:,2] >= bin_edges[2][k],
                                                    X_train[:,2] < bin_edges[2][k + 1]])
            bin_mask_val = np.logical_and.reduce([X_val[:,0] >= bin_edges[0][i],
                                                  X_val[:,0] < bin_edges[0][i + 1],
                                                  X_val[:,1] >= bin_edges[1][j],
                                                  X_val[:,1] < bin_edges[1][j + 1],
                                                  X_val[:,2] >= bin_edges[2][k],
                                                  X_val[:,2] < bin_edges[2][k + 1]])
            bin_masks_train.append(bin_mask_train)
            bin_masks_val.append(bin_mask_val)

            # Apply the filters to train and validation data
            bin_mask_train = np.logical_and.reduce(bin_masks_train)
            bin_mask_val = np.logical_and.reduce(bin_masks_val)

            X_train_bin = X_train[bin_mask_train]
            y_train_bin = y_train[bin_mask_train]
            X_val_bin = X_val[bin_mask_val]
            y_val_bin = y_val[bin_mask_val]

            # Evaluate the model on the current bin and save the results
            train_metrics_bin, val_metrics_bin = -1, -1 #set a default value of -1
            if X_train_bin.shape[0] != 0:
              train_metrics_bin = nnetwork.evaluate(X_train_bin, y_train_bin, verbose=0)
            if X_val_bin.shape[0] != 0:
              val_metrics_bin = nnetwork.evaluate(X_val_bin, y_val_bin, verbose=0)            

            train_metrics.append(train_metrics_bin)
            val_metrics.append(val_metrics_bin)

      # Append metrics for the current epoch
      train_metrics_per_bin.append(train_metrics)
      val_metrics_per_bin.append(val_metrics)

    # Transform the metrics per bin into numpy arrays
    train_metrics_per_bin = np.array(train_metrics_per_bin)
    val_metrics_per_bin = np.array(val_metrics_per_bin)

    # Plot the train-validation metrics for each bin and epoch
    for k, g_plot in enumerate(bin_edges[2][:-1]):
      fig, axs = plt.subplots(n_bins, n_bins, figsize=(12, 12))   #, sharex=True, sharey=True
      for i, B_plot in enumerate(bin_edges[0][:-1]):
        for j, D_plot in enumerate(bin_edges[1][:-1]):
          train_metrics_bin = train_metrics_per_bin[:, i * n_bins**2 + j*n_bins + k]
          val_metrics_bin = val_metrics_per_bin[:, i * n_bins**2 + j*n_bins + k]

          # Plot train-validation metrics
          axs[n_bins-i-1, j].plot(train_metrics_bin, label='Train')
          axs[n_bins-i-1, j].plot(val_metrics_bin, label='Validation')
          axs[n_bins-i-1, j].set_title(f'B = {B_plot:.1f}, D = {D_plot:.2f}')
          axs[n_bins-i-1, j].set_xlabel('Epoch')
          axs[n_bins-i-1, j].set_ylabel('Metric')
          axs[n_bins-i-1, j].legend()

      plt.tight_layout()
      plt.savefig(f'temp/training_history_g_{g_plot}.png')

    np.savez('binned_train_val.npy', array1=train_metrics_per_bin, array2=val_metrics_per_bin)
    print('Successfully completed Neural Network training.')

  # Retrieve the loss name
  try:
    loss_name = nnetwork.loss.__name__
  except AttributeError:
    loss_name = nnetwork.loss.name

  with open(os.path.join('data','train_summary.pkl'), 'rb') as f:
      rf_summary, nn_summary = pickle.load(f)
      
  # Save the training summary
  if (mode=='rf' or mode=='all'):
    rf_summary = "".join(['\n\n***RANDOM FOREST***',
                          '\nn_estimators = {}'.format(rforest.get_params()['n_estimators']),
                          '\nmax_features = {}'.format(rforest.get_params()['max_features']),
                          '\nmax_samples = {}'.format(rforest.get_params()['max_samples']),
                          '\nmin_samples_leaf = {}'.format(rforest.get_params()['min_samples_leaf']),
                          '\nmin_samples_split = {}'.format(rforest.get_params()['min_samples_split'])])
  if (mode=='nn' or mode=='all'):
    nn_summary = "".join(['\n\n***NEURAL NETWORK***',
                          '\nloss_name = {}'.format(loss_name),
                          '\nn_epochs = {}'.format(n_epochs)])

    for key, value in hp.items():
      nn_summary += f"\n{key} = {value}"

  with open(os.path.join('data','train_summary.pkl'), 'wb') as f:
      pickle.dump([rf_summary, nn_summary], f)

  return 