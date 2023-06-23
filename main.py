# PREDICTION OF BIOMASS/SOIL DEPTH TRANSITION WITH VARYING GRAZING PRESSURE

# Import the necessary libraries
print('Importing libraries and modules...')
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import joblib as jb
import matplotlib.pyplot as plt
from keras.models import load_model
from modules.train_models import train_models
from modules.test_eval import test_eval
from modules.train_eval import train_eval
from modules.system_evolution import system_evolution
from modules.surface_plots import surface_plots
from modules.colormesh_plots import colormesh_plots
from modules.tipping_evolution import tipping_evolution
print('Successfully imported libraries and modules.')

# Set which functionalities to use
remove_outliers = False     # False, True
model_training = 'all'      # False, 'rf', 'nn' or 'all'.
model_evaluation = 'all'    # False, 'train', 'test', 'all'
plots = ['surface']         # ['surface', 'colormesh', 'tipping']
system_ev = [0,1,'val_data_lin']              # [0,1,2,'val_data_sin','val_data_lin']

run_summary = "".join(['***MODULES***',
                       '\nremove_outliers = {}'.format(remove_outliers),
                       '\nmodel_training = {}'.format(model_training),
                       '\nmodel_evaluation = {}'.format(model_evaluation),
                       '\nsystem_ev = {}'.format(system_ev),
                       '\nplots = {}'.format(plots)])

# Record starting run time
start_time = time.time()

# Load the data
data_file = 'gd_data.pkl'
print('Loading and formatting data...')
with open(os.path.join('data', data_file), 'rb') as f:
    B,D,g,dB_dt,dD_dt = pickle.load(f)

# Save the necessary data for the system evolution
X_ev = system_ev
for i, element in enumerate(X_ev):
  if isinstance(element, int):
    X_ev[i] = [B[:,element], D[:,element], g[:,element]]

# Define input and output variables and delete unnecessary data
X = np.column_stack((B.flatten('F'),D.flatten('F'),g.flatten('F')))
y = np.column_stack((dB_dt.flatten('F'),dD_dt.flatten('F')))
del B,D,g,dB_dt,dD_dt

n_samples = X.shape[0]

# Split between training and test data and delete unnecessary data
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    shuffle=False)
del X, y

# Split between training and validation data
val_size = 0.2
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=val_size/(1-test_size),
                                                  shuffle=True, random_state=123)
# Add the data characteristics to the summary
run_summary += "".join(['\n\n***DATA***',
                        '\nn_samples = {}'.format(n_samples),
                        '\ntest_size = {}'.format(test_size),
                        '\nval_size = {}'.format(val_size)])
# Remove outliers if requested
if remove_outliers:

  # Find the outliers
  q1, q3 = np.percentile(y_train, [25, 75])
  lower_bound = q1 - (2000 * (q3 - q1))
  upper_bound = q3 + (2000 * (q3 - q1))
  outliers = np.any((y_train < lower_bound) | (y_train > upper_bound), axis=1)

  # Plot the data highliting the removed outliers
  fig, ax = plt.subplots(figsize=(15,9))
  ax.plot(y_train[~outliers][:,0], y_train[~outliers][:,1], '.k')
  ax.plot(y_train[outliers][:,0], y_train[outliers][:,1], '.r')
  ax.set_xlabel('Biomass rate of change')
  ax.set_ylabel('Soil Depth rate of change')
  plt.savefig(os.path.join('temp', 'outlier_detection.png'))

  # Remove the outliers from the dataset
  X_train, y_train = X_train[~outliers], y_train[~outliers]

  # Add to the summary and print the number of removed outliers 
  run_summary += '\noutliers_removed = {}'.format(np.sum(outliers))
  print(f'\n\n***\n{np.sum(outliers)} outliers removed.\n***\n\n')
                        
print('Successfully loaded and formatted data...')

# Train the models if specified
if model_training != False:
  train_models(X_train, X_val, y_train, y_val, mode=model_training)

# Load the models
nnetwork = load_model(os.path.join('data', 'nn_model.h5'), compile=False)
rf_params = jb.load(os.path.join('data', 'rf_model.joblib'))

# Define a variant of the random forest that uses the trees median to predict
class MedianRandomForestRegressor(RandomForestRegressor):
  def predict(self, X):
      all_predictions = []
      for tree in self.estimators_:
          all_predictions.append(tree.predict(X))
      return np.median(all_predictions, axis=0)
rforest = MedianRandomForestRegressor()
rforest.__dict__ = rf_params.__dict__

# Load the training summary
with open(os.path.join('data','train_summary.pkl'), 'rb') as f:
    rf_summary, nn_summary = pickle.load(f)

# Evaluate the training data specified in model_evaluation
if (model_evaluation=='train' or model_evaluation=='all'):
  train_summary = train_eval(rforest, nnetwork, X_train, y_train, X_val, y_val)
  rf_summary += train_summary[0]
  nn_summary += train_summary[1]

# Evaluate the test data if set to True
if (model_evaluation=='test' or model_evaluation=='all'):
  test_summary = test_eval(nnetwork, rforest, X_test, y_test)
  rf_summary += test_summary[0]
  nn_summary += test_summary[1]

# Add the model summaries to the run summary
run_summary += rf_summary
run_summary += nn_summary

# Plot the predicted rate of change for B and D at critical g if in the plots list
if 'surface' in plots:
  run_summary += surface_plots(nnetwork, rforest)

# Plot colormeshes related to the observations available if in the plots list
if 'colormesh' in plots:
  run_summary += colormesh_plots(X_train, y_train)

# Plot the system evolutionat the tipping point if in the plots list
if 'tipping' in plots:
  run_summary += tipping_evolution(nnetwork)

# Make a prediction of the evolution of the system for each simulation in X_ev
ev_summary = '\n\n***SYSTEM EVOLUTION***'
for i,sim in enumerate(X_ev):
  ev_summary += system_evolution(nnetwork, rforest, sim, i)
run_summary += ev_summary

# Print execution time
end_time = time.time()
execution_time = (end_time - start_time)/60
print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time),
      '\nEnd time: {}'.format(time.ctime(end_time)))

run_summary += "".join(['\n\n***\nExecution time: {:.3g} minutes.'.format(execution_time),
                        '\nEnd time: {}\n***'.format(time.ctime(end_time))])

with open(os.path.join('results', 'run_summary.txt'), 'w') as f:
    f.write(run_summary)