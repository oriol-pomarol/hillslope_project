import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt
import os
import joblib as jb
from .data_preparation import subset_mask_stratified
from .surface_plots import surface_plots

def train_models(processed_data, mode='all', add_train_vars=[None]*3):

  # Unpack the data
  X_train, X_val, y_train, y_val, _, _ = processed_data
  w_train, len_eq_train, len_eq_val = add_train_vars

  # # Shuffle the training data
  # shuffled_indices = np.arange(len(X_train))
  # np.random.shuffle(shuffled_indices)
  # X_train = np.squeeze(X_train[shuffled_indices])
  # y_train = np.squeeze(y_train[shuffled_indices])
  # w_train = np.squeeze(w_train[shuffled_indices])

  if (mode=='rf' or mode=='all'):
    # Start the random forest model training
    print('Starting Random Forest training...')
    rforest = RandomForestRegressor(n_estimators = 200,
                                    max_features = 'sqrt',
                                    max_samples = 0.4,
                                    min_samples_leaf = 2,
                                    min_samples_split = 20)
    train_rf_start = time.time()
    rforest.fit(X_train, y_train, sample_weight=w_train)

    train_rf_end = time.time()
    train_rf_time = (train_rf_end - train_rf_start)/60
    print('RF training time: {:.3g} minutes.'.format(train_rf_time))

    # Save the model
    jb.dump(rforest, os.path.join('data','rf_model.joblib')) 
    print('Successfully completed Random Forest training.')

  if (mode=='nn' or mode=='all'):
    print('Starting Neural Network training...')

    # Set a random seed for tensorflow and numpy to ensure reproducibility
    tf.random.set_seed(10)
    np.random.seed(10)

    # Obtain the standard deviations of the training data
    dB_dt_std = np.std(y_train[:,0])
    dD_dt_std = np.std(y_train[:,1])

    #define a loss function
    def custom_mae(y_true, y_pred):
      loss = y_pred - y_true
      loss = loss / [dB_dt_std, dD_dt_std]
      loss = K.abs(loss)
      loss = K.sum(loss, axis=1) 
      return loss

    # Set the hyperparameters
    hp = {'units':[9, 27, 81, 162, 324, 648, 1296], 'act_fun':'relu',
          'learning_rate':1E-5, 'batch_size':128, 'l1_reg':1e-5, 'n_epochs':200}

    # Define what hyperparameter to tune and its values
    tuning_hp_name = 'w_eq'
    tuning_hp_vals = []

    if tuning_hp_vals:
      print('Starting hyperparameter tuning...')
      best_hp = hp_tuning(train_val_data, add_train_vars, hp, custom_mae,
                          tuning_hp_name, tuning_hp_vals)
      # Apply the best hyperparameter
      if tuning_hp_name == 'w_eq':
        # Generate the weights for the equilibrium and jumps data
        length_ratio = len_eq_train/(len(X_train) - len_eq_train)
        w_train = np.concatenate((best_hp*np.ones(len_eq_train)*length_ratio,
                                  (1-best_hp)*np.ones(len(X_train) - len_eq_train)))
      else:
        hp[tuning_hp_name] = best_hp
      print('Successfully tuned hyperparameters.')

    # Define the model
    nnetwork = keras.Sequential()
    for n_units in hp['units']:
      nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=hp['act_fun'],
                                        kernel_regularizer=keras.regularizers.l1(hp['l1_reg'])))
    nnetwork.add(keras.layers.Dense(2, activation='linear',
                                    kernel_regularizer=keras.regularizers.l1(hp['l1_reg'])))
    # Compile and fit the model
    nnetwork.compile(optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate']), loss=custom_mae)
    train_nn_start = time.time()

    # Add the separate validation sets if they are provided
    if len_eq_val is None:
      history = nnetwork.fit(X_train, y_train, epochs = hp['n_epochs'], validation_data = (X_val, y_val),
                               batch_size = hp['batch_size'], sample_weight = w_train)
    else:
      X_val_eq = X_val[:len_eq_val]
      y_val_eq = y_val[:len_eq_val]
      X_val_jp = X_val[len_eq_val:]
      y_val_jp = y_val[len_eq_val:]
      history = AdditionalValidationSets([(X_val_eq, y_val_eq, 'val_loss_eq'), (X_val_jp, y_val_jp, 'val_loss_jp')])
      nnetwork.fit(X_train, y_train, epochs = hp['n_epochs'], validation_data = (X_val, y_val),
                   batch_size = hp['batch_size'], sample_weight = w_train, callbacks=[history])
    
    # Save the history as a pandas dataframe
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join('results','training_history.csv'))
    
    # Plot the MSE history of the training
    plt.figure()
    for key in history.history.keys():
      plt.plot(history_df[key], label=key)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Custom MSE')
    plt.savefig(os.path.join('results','training_history.png'))

    train_nn_end = time.time()
    train_nn_time = (train_nn_end - train_nn_start)/60
    print('NN training time: {:.3g} minutes.'.format(train_nn_time))

    nnetwork.save(os.path.join('data', 'nn_model.h5'))
    print('Successfully completed Neural Network training.')

    # Retrieve the loss name
    try:
      loss_name = nnetwork.loss.__name__
    except AttributeError:
      loss_name = nnetwork.loss.name

  # If training only one model, load the parameters from the other model
  if (mode=='rf' or mode=='nn'):
    with open(os.path.join('data','train_summary.pkl'), 'rb') as f:
        rf_summary, nn_summary = pickle.load(f)
      
  # Save the training summary
  if (mode=='rf' or mode=='all'):
    rf_summary = "".join(['\n\n***MODEL TRAINING***',
                          '\n\nRANDOM FOREST:',
                          '\ntrain_rf_time = {:.1f} minutes'.format(train_rf_time),
                          '\nn_estimators = {}'.format(rforest.get_params()['n_estimators']),
                          '\nmax_features = {}'.format(rforest.get_params()['max_features']),
                          '\nmax_samples = {}'.format(rforest.get_params()['max_samples']),
                          '\nmin_samples_leaf = {}'.format(rforest.get_params()['min_samples_leaf']),
                          '\nmin_samples_split = {}'.format(rforest.get_params()['min_samples_split'])])
  if (mode=='nn' or mode=='all'):
    nn_summary = "".join(['\n\nNEURAL NETWORK:',
                          '\ntrain_nn_time = {:.1f} minutes'.format(train_nn_time),
                          '\nloss_name = {}'.format(loss_name)])

    for key, value in hp.items():
      nn_summary += f"\n{key} = {value}"

  with open(os.path.join('data','train_summary.pkl'), 'wb') as f:
      pickle.dump([rf_summary, nn_summary], f)

  return

def hp_tuning(train_val_data, add_train_vars, hp, loss, tuning_hp_name, tuning_hp_vals):

  # Unpack the data
  X_train, X_val, y_train, y_val = train_val_data
  w_train, len_eq_train, len_eq_val = add_train_vars

  # Create empty lists to store the losses
  losses = []
  losses_eq = []
  losses_jp = []

  # Make a mask to subset of the data for tuning, conserving the eq/jp ratio
  tuning_factor = 0.1
  tuning_mask, len_eq_tuning, len_jp_tuning = \
    subset_mask_stratified(tuning_factor, len(X_train), len_eq_train)

  # Subset the data for tuning
  X_tuning = X_train[tuning_mask]
  y_tuning = y_train[tuning_mask]
  if w_train is not None:
    w_tuning = w_train[tuning_mask]

  # Test each hyperparameter value
  for i, value in enumerate(tuning_hp_vals):
    print(f'Testing hp {i+1} of {len(tuning_hp_vals)}...')

    # Change the hyperparameter to the tuning value
    if tuning_hp_name == 'w_eq':
      # Generate the weights for the equilibrium and jumps data
      length_ratio = len_jp_tuning/len_eq_tuning
      w_tuning = np.concatenate((value*np.ones(len_eq_tuning)*length_ratio,
                                (1-value)*np.ones(len_jp_tuning)))
    else:
      hp[tuning_hp_name] = value

    # Define the model
    nnetwork = keras.Sequential()
    for n_units in hp['units']:
      nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=hp['act_fun'],
                                        kernel_regularizer=keras.regularizers.l1(hp['l1_reg'])))
    nnetwork.add(keras.layers.Dense(2, activation='linear',
                                    kernel_regularizer=keras.regularizers.l1(hp['l1_reg'])))

    # Compile and fit the model
    nnetwork.compile(optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate']), loss=loss)
    history = nnetwork.fit(X_tuning, y_tuning, epochs = hp['n_epochs'], validation_data = (X_val, y_val), 
                          batch_size = hp['batch_size'], sample_weight = w_tuning)
    
    # Make a surface plot of the model
    surface_plots(nnetwork, name=f'{tuning_hp_name}_{value}')
    
    if len_eq_val is not None:
      # Split the validation loss into equilibrium and jumps
      X_val_eq = X_val[:len_eq_val]
      y_val_eq = y_val[:len_eq_val]
      X_val_jp = X_val[len_eq_val:]
      y_val_jp = y_val[len_eq_val:]

      # Evaluate the model on the validation sets
      val_loss_eq = nnetwork.evaluate(X_val_eq, y_val_eq)
      val_loss_jp = nnetwork.evaluate(X_val_jp, y_val_jp)
      losses_eq.append(val_loss_eq)
      losses_jp.append(val_loss_jp)

    losses.append(history.history['val_loss'][-1])

  # Save a csv with each model's loss and hp
  tuning_res_dict = {tuning_hp_name:tuning_hp_vals, 'loss':losses}
  if len_eq_val is not None:
    tuning_res_dict['loss_eq'] = losses_eq
    tuning_res_dict['loss_jp'] = losses_jp
  pd.DataFrame(tuning_res_dict).to_csv(os.path.join('results','hp_tuning.csv'))

  # Select the best hyperparameter and retrain the model
  best_hp = tuning_hp_vals[np.argmin(losses)]
  print('Best model has {} = {:.1g}.'.format(tuning_hp_name, best_hp))
  return best_hp

# Source: https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras
# Should rewrite this for my specific problem
from keras.callbacks import Callback
class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [3, 4]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)
            if isinstance(results, float):
                results = [results]  # Convert single float to a list

            for metric, result in zip(self.model.metrics_names,results):
                valuename = validation_set_name + '_' + metric
                self.history.setdefault(valuename, []).append(result)