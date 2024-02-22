# PREDICTION OF BIOMASS/SOIL DEPTH TRANSITION WITH VARYING GRAZING PRESSURE

# Import the necessary libraries
print('Importing libraries and modules...')
import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib as jb
from keras.models import load_model
from modules.data_preparation import data_preparation
from modules.train_models import train_models
from modules.test_eval import test_eval
from modules.train_eval import train_eval
from modules.system_evolution import system_evolution
from modules.surface_plots import surface_plots
from modules.colormesh_plots import colormesh_plots
from modules.tipping_evolution import tipping_evolution
from config import main as cfg
from config import paths
print('Successfully imported libraries and modules.')

# Record starting run time
start_time = time.time()

# Initialize the summary
run_summary = ""

# Prepare the data for training
print('Preparing the data...')
data_summary = data_preparation()
run_summary += data_summary
print('Successfully prepared the data...')

# Train the models if specified
if cfg.model_training != 'none':
  train_models(cfg.model_training)

# Load the models
nnetwork = load_model(paths.models / 'nn_model.h5', compile=False)
rf_params = jb.load(paths.models / 'rf_model.joblib')

# Define a variant of the random forest that uses the trees median to predict
class MedianRandomForestRegressor(RandomForestRegressor):
  def predict(self, X):
    return np.median([tree.predict(X) for tree in self.estimators_], axis=0)
rforest = MedianRandomForestRegressor()
rforest.__dict__ = rf_params.__dict__

# Load the training summary
with open(paths.temp_data / 'train_summary.pkl', 'rb') as f:
    rf_summary, nn_summary = pickle.load(f)

# Evaluate the training data specified in model_evaluation
if (cfg.model_evaluation=='train' or cfg.model_evaluation=='all'):
  train_summary = train_eval(rforest, nnetwork)
  rf_summary += train_summary[0]
  nn_summary += train_summary[1]

# Evaluate the test data if set to True
if (cfg.model_evaluation=='test' or cfg.model_evaluation=='all'):
  test_summary = test_eval(nnetwork, rforest)
  rf_summary += test_summary[0]
  nn_summary += test_summary[1]

# Add the model summaries to the run summary
run_summary += rf_summary
run_summary += nn_summary

# Plot the predicted rate of change for B and D at critical g if in the plots list
if 'surface' in cfg.plots:
  run_summary += surface_plots(nnetwork, name='nn')
  surface_plots(rforest, name='rf')

# Plot colormeshes related to the observations available if in the plots list
if 'colormesh' in cfg.plots:
  run_summary += colormesh_plots()

# Plot the system evolutionat the tipping point if in the plots list
if 'tipping' in cfg.plots:
  run_summary += tipping_evolution(nnetwork)

# Make a prediction of the evolution of the system for each simulation in X_ev
ev_summary = '\n\n***SYSTEM EVOLUTION***'
for i,sim in enumerate(cfg.fwd_sim):
  ev_summary += system_evolution(nnetwork, rforest, sim, i)
run_summary += ev_summary

# Print execution time
end_time = time.time()
execution_time = (end_time - start_time)/60
print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time),
      '\nEnd time: {}'.format(time.ctime(end_time)))

run_summary += "".join(['\n\n***\nExecution time: {:.3g} minutes.'.format(execution_time),
                        '\nEnd time: {}\n***'.format(time.ctime(end_time))])

with open(paths.outputs / 'run_summary.txt', 'w') as f:
    f.write(run_summary)