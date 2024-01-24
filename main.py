# PREDICTION OF BIOMASS/SOIL DEPTH TRANSITION WITH VARYING GRAZING PRESSURE

# Import the necessary libraries
print('Importing libraries and modules...')
import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import joblib as jb
from keras.models import load_model
from modules.data_generation import data_generation
from modules.data_formatting import data_formatting
from modules.train_models import train_models
from modules.test_eval import test_eval
from modules.train_eval import train_eval
from modules.system_evolution import system_evolution
from modules.surface_plots import surface_plots
from modules.colormesh_plots import colormesh_plots
from modules.tipping_evolution import tipping_evolution
print('Successfully imported libraries and modules.')

# Set which functionalities to use
model_training = 'all'      # False, 'rf', 'nn' or 'all'.
model_evaluation = 'all'    # False, 'train', 'test', 'all'
plots = ['surface']         # ['surface', 'colormesh', 'tipping']
system_ev = []              # [0,1,2,'val_data_sin','val_data_lin']

run_summary = "".join(['***MODULES***',
                       '\nmodel_training = {}'.format(model_training),
                       '\nmodel_evaluation = {}'.format(model_evaluation),
                       '\nsystem_ev = {}'.format(system_ev),
                       '\nplots = {}'.format(plots)])

# Record starting run time
start_time = time.time()

# Load and preprocess/generate the data
print('Generating data...')
gen_summary, jp_eq_data = data_generation()
run_summary += gen_summary
print('Successfully generated data...')

# Prepare the data for training
print('Formatting data...')
data_summary, train_val_data, test_data, add_train_vars = \
  data_formatting(jp_eq_data)
run_summary += data_summary
print('Successfully formatted data...')

# Train the models if specified
if model_training != False:
  train_models(train_val_data, add_train_vars,
               model_training)

# Load the models
nnetwork = load_model(os.path.join('data', 'nn_model.h5'), compile=False)
rf_params = jb.load(os.path.join('data', 'rf_model.joblib'))

# Define a variant of the random forest that uses the trees median to predict
class MedianRandomForestRegressor(RandomForestRegressor):
  def predict(self, X):
    return np.median([tree.predict(X) for tree in self.estimators_], axis=0)
rforest = MedianRandomForestRegressor()
rforest.__dict__ = rf_params.__dict__

# Load the training summary
with open(os.path.join('data','train_summary.pkl'), 'rb') as f:
    rf_summary, nn_summary = pickle.load(f)

# Evaluate the training data specified in model_evaluation
if (model_evaluation=='train' or model_evaluation=='all'):
  train_summary = train_eval(rforest, nnetwork, train_val_data)
  rf_summary += train_summary[0]
  nn_summary += train_summary[1]

# Evaluate the test data if set to True
if (model_evaluation=='test' or model_evaluation=='all'):
  test_summary = test_eval(nnetwork, rforest, test_data)
  rf_summary += test_summary[0]
  nn_summary += test_summary[1]

# Add the model summaries to the run summary
run_summary += rf_summary
run_summary += nn_summary

# Plot the predicted rate of change for B and D at critical g if in the plots list
if 'surface' in plots:
  run_summary += surface_plots(nnetwork, name='nn')
  surface_plots(rforest, name='rf')

# Plot colormeshes related to the observations available if in the plots list
if 'colormesh' in plots:
  run_summary += colormesh_plots(train_val_data)

# Plot the system evolutionat the tipping point if in the plots list
if 'tipping' in plots:
  run_summary += tipping_evolution(nnetwork)

# # Make a prediction of the evolution of the system for each simulation in X_ev
# ev_summary = '\n\n***SYSTEM EVOLUTION***'
# for i,sim in enumerate(X_ev):
#   ev_summary += system_evolution(nnetwork, rforest, sim, i)
# run_summary += ev_summary

# Print execution time
end_time = time.time()
execution_time = (end_time - start_time)/60
print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time),
      '\nEnd time: {}'.format(time.ctime(end_time)))

run_summary += "".join(['\n\n***\nExecution time: {:.3g} minutes.'.format(execution_time),
                        '\nEnd time: {}\n***'.format(time.ctime(end_time))])

with open(os.path.join('results', 'run_summary.txt'), 'w') as f:
    f.write(run_summary)