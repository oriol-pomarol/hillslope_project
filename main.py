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
import matplotlib.ticker as tck
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
model_training = 'all'      # False, 'rf', 'nn' or 'all'.
model_evaluation = 'all'    # False, 'train', 'test', 'all'
plots = []         # ['surface', 'colormesh', 'tipping']
system_ev = []              # [0,1,2,'val_data_sin','val_data_lin']

run_summary = "".join(['***MODULES***',
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
    B_input,D_input,g,_,_ = pickle.load(f)
    
# # Subset the first 5 columns (for test purposes)
# B_input = B_input[:, :5]
# D_input = D_input[:, :5]
# g = g[:, :5]

def gen_g_hy(n_steps=0):
  g = np.ones(n_steps) * np.random.uniform(0, 3)
  prob_new_g = 0.01 # probability of setting a new random g value
  for step in range(1, n_steps):
    g[step] = g[step-1]
    if np.random.choice([True, False], p=[prob_new_g, 1 - prob_new_g]):
      g[step] = np.random.uniform(0, 3)
  return g

# Define some run parameters
Bo = B_input[0]   # initial value of B
Do = D_input[0]   # initial value of D
dt = 0.5        # time step, 7/365 in paper, 0.5 for stability in results
n_steps = len(B_input)
n_years = dt*n_steps   # maximum number of years to run, 20000 in paper
prob_new_B = 0.01 # probability of setting a new random B value
prob_new_D = 0.01 # probability of setting a new random D value
for i in range(g.shape[1]):
  g[:, i] = gen_g_hy(n_steps=n_steps)

# Define the physical parameters
r, c, i, d, s = 2.1, 2.9, -0.7, 0.04, 0.4 
Wo, a, Et, Eo, k, b, C = 5e-4, 4.02359478109, 0.021, 0.084, 0.05, 0.28, 1e-4
alpha = np.log(Wo/C)/a

# Define the function that computes dB/dt and dD/dt
def dX_dt(B,D,g):
  dB_dt_step = (1-(1-i)*np.exp(-1*D/d))*(r*B*(1-B/c))-g*B/(s+B)
  dD_dt_step = Wo*np.exp(-1*a*D)-np.exp(-1*B/b)*(Et+np.exp(-1*D/k)*(Eo-Et))-C
  return dB_dt_step, dD_dt_step

# Generate the time sequence
t = np.linspace(0, n_years, n_steps)

# Initialize B and D
B_steps = np.ones_like(B_input) * Bo
D_steps = np.ones_like(D_input) * Do

# Initialize dB/dt and dD/dt
dB_dt_steps = np.ones_like(B_input)
dD_dt_steps = np.ones_like(D_input)

# Create a jumps array to keep track of when they happen
jumps = np.full((B_input.shape[0], B_input.shape[1]), False)
jumps[0, :] = True

# Allow the system to evolve
for step in range(1,n_steps):
  
  # Compute the derivatives
  steps_slopes = dX_dt(B_steps[step-1], D_steps[step-1], g[step-1])
  dB_dt_steps[step-1], dD_dt_steps[step-1] = steps_slopes

  # Compute the new values, forced to be above 0 and below their theoretical max
  B_steps[step] = np.clip(B_steps[step-1] + steps_slopes[0]*dt, 0.0, c)
  D_steps[step] = np.clip(D_steps[step-1] + steps_slopes[1]*dt, 0.0, alpha)

  # Add a random chance to set a new random B value
  jump_B = np.random.choice([True, False], size=len(B_steps[step]), p=[prob_new_B, 1 - prob_new_B])
  B_steps[step][jump_B] = np.random.uniform(0, c, size=len(B_steps[step][jump_B]))
  jumps[step][jump_B] = True

  # Add a random chance to set a new random D value
  jump_D = np.random.choice([True, False], size=len(D_steps[step]), p=[prob_new_D, 1 - prob_new_D])
  D_steps[step][jump_D] = np.random.uniform(0, alpha, size=len(D_steps[step][jump_D]))
  jumps[step][jump_D] = True
    
dB_dt_steps[-1], dD_dt_steps[-1] = dX_dt(B_steps[-1], D_steps[-1], g[-1])
jumps_shifted = np.roll(jumps, -1, axis=0)

# Plot D(t), B(t) and g(t) for the first simulation
fig, axs = plt.subplots(3, 1, figsize = (10,7.5))

axs[0].plot(t, D_steps[:,0], '-b', label = 'Minimal model')
axs[0].set_ylim(0)
axs[0].set_ylabel('soil thickness')
axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
axs[0].tick_params(axis="both", which="both", direction="in", 
                        top=True, right=True)

axs[1].plot(t, B_steps[:,0], '-b')
axs[1].set_ylim(0)
axs[1].set_ylabel('biomass')
axs[1].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
axs[1].tick_params(axis="both", which="both", direction="in", 
                        top=True, right=True)

axs[2].plot(t, g[:,0], '-b')
axs[2].set_ylim(0)
axs[2].set_ylabel('grazing pressure')
axs[2].set_xlabel('time (years)')
axs[2].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
axs[2].tick_params(axis="both", which="both", direction="in", 
                        top=True, right=True)

fig.patch.set_alpha(1)
plt.setp(axs, xlim=(0, n_years))
plt.savefig(f'results/train_sim_0.png')

# Mask observations where B, D, dB_dt and dD_dt are all zero
zero_values = (B_steps == 0.0) & (D_steps == 0.0)

print(f"{np.sum(zero_values)} boundary values found.")

# Save the results as the new training data
B_input = B_steps
D_input = D_steps
g_input = g
dB_dt = dB_dt_steps
dD_dt = dD_dt_steps

del B_steps, D_steps, dB_dt_steps, dD_dt_steps

# Save the necessary data for the system evolution
X_ev = system_ev
for i, element in enumerate(X_ev):
  if isinstance(element, int):
    X_ev[i] = [B_input[:,element], D_input[:,element], g[:,element]]

# Define input and output variables and delete zero values
X = np.column_stack((B_input.flatten('F'),D_input.flatten('F'),g_input.flatten('F')))
y = np.column_stack((dB_dt.flatten('F'),dD_dt.flatten('F')))
X = X[~(zero_values.flatten('F') | jumps_shifted.flatten('F'))]
y = y[~(zero_values.flatten('F') | jumps_shifted.flatten('F'))]
del B_input,D_input,g,dB_dt,dD_dt

shape_data = X.shape
print("Final data shape:", shape_data)

# Split between training and test data and delete unnecessary data
test_size = 0.2
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    shuffle=False)
del X, y

# Split between training and validation data
val_size = 0.1
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, 
                                                  test_size=val_size/(1-test_size),
                                                  shuffle=True, random_state=123)
del X_train_val, y_train_val
train_samples = X_train.shape[0]
# Add the data characteristics to the summary
run_summary += "".join(['\n\n***DATA***',
                        '\nshape_data = {}'.format(shape_data),
                        '\ntrain_samples = {}'.format(train_samples),
                        # '\nrmv_samples = {}'.format(np.sum(zero_values)),
                        '\ntest_size = {}'.format(test_size),
                        '\nval_size = {}'.format(val_size),
                        '\nprob_new_B = {}'.format(prob_new_B),
                        '\nprob_new_D = {}'.format(prob_new_D)])
                        
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