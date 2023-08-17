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

def train_models(X_train, X_val, y_train, y_val, mode='all'):

  if (mode=='rf' or mode=='all'):
    # Start the random forest model training
    print('Starting Random Forest training...')
    rforest = RandomForestRegressor(n_estimators = 100, max_features = 'sqrt', 
      max_samples = 0.4, min_samples_leaf = 2, min_samples_split = 20)
    rforest.fit(X_train, y_train)
    jb.dump(rforest, os.path.join('data','rf_model.joblib')) 
    print('Successfully completed Random Forest training.')

  if (mode=='nn' or mode=='all'):

    # Set a random seed for tensorflow
    tf.random.set_seed(10)

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

    hp = {'units':[9, 27, 81, 162, 324, 648, 1296], 'act_fun':'relu', 'learning_rate':1E-5, 'batch_size':64}

    # Define the model
    nnetwork = keras.Sequential()
    for n_units in hp['units']:
      nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=hp['act_fun']))
    nnetwork.add(keras.layers.Dense(2, activation='linear', kernel_regularizer=keras.regularizers.l1(0.01)))

    # Compile and fit the model
    n_epochs = 7
    print('Starting Neural Network training...')
    nnetwork.compile(optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate']), loss=custom_mse)
    history = nnetwork.fit(X_train, y_train, epochs = n_epochs, validation_data = (X_val, y_val), 
                           batch_size = hp['batch_size'])

    # Plot the MSE history of the training
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Custom MSE')
    plt.savefig(os.path.join('results','training_history.png'))

    df = pd.DataFrame({'loss':history.history['loss'], 'val_loss':history.history['val_loss']})
    df.to_csv(os.path.join('results','training_history.csv'))
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