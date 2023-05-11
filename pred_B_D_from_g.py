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
print('Successfully imported libraries and modules.')

# Set which functionalities to use
train_mod = False
train_ev = False
test_ev = False
surface_pl = True
colormesh_pl = False
system_ev = [] #0,1,2,'val_data_sin','val_data_lin'

run_summary = "".join(['***MODULES***',
                       '\ntrain_mod = {}'.format(train_mod),
                       '\ntrain_ev = {}'.format(train_ev),
                       '\ntest_ev = {}'.format(test_ev),
                       '\nsystem_ev = {}'.format(system_ev),
                       '\nsurface_pl = {}'.format(surface_pl)])

# Record starting run time
start_time = time.time()

# Load the data
data_file = 'data.pkl'
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
# Remove outliers
q1, q3 = np.percentile(y_train, [25, 75])
lower_bound = q1 - (2000 * (q3 - q1))
upper_bound = q3 + (2000 * (q3 - q1))
outliers = np.any((y_train < lower_bound) | (y_train > upper_bound), axis=1)

fig, ax = plt.subplots(figsize=(15,9))
ax.plot(y_train[~outliers][:,0], y_train[~outliers][:,1], '.k')
ax.plot(y_train[outliers][:,0], y_train[outliers][:,1], '.r')
ax.set_xlabel('Biomass rate of change')
ax.set_ylabel('Soil Depth rate of change')
plt.savefig(os.path.join('temp', 'outlier_detection.png'))

X_train, y_train = X_train[~outliers], y_train[~outliers]

print(f'\n\n***\n{np.sum(outliers)} outliers removed.\n***\n\n')

run_summary += "".join(['\n\n***DATA***',
                        '\nn_samples = {}'.format(n_samples),
                        '\ntest_size = {}'.format(test_size),
                        '\nval_size = {}'.format(val_size),
                        '\noutliers_removed = {}'.format(np.sum(outliers))])
print('Successfully loaded and formatted data...')

# Train the models if set to True
if train_mod != False:
  train_models(X_train, X_val, y_train, y_val, mode=train_mod)

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

# Evaluate the training data if set to True
if train_ev:
  train_summary = train_eval(rforest, nnetwork, X_train, y_train, X_val, y_val)
  rf_summary += train_summary[0]
  nn_summary += train_summary[1]

# Evaluate the test data if set to True
if test_ev:
  test_summary = test_eval(nnetwork, rforest, X_test, y_test)
  rf_summary += test_summary[0]
  nn_summary += test_summary[1]

# Add the model summaries to the run summary
run_summary += rf_summary
run_summary += nn_summary

# Plot the predicted rate of change for B and D at critical g if set to True
if surface_pl:
  run_summary += surface_plots(nnetwork, rforest)

# Plot the predicted rate of change for B and D at critical g if set to True
if colormesh_pl:
  run_summary += colormesh_plots(X_train, y_train)

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