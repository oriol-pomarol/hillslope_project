from pathlib import Path
import numpy as np
from config import data_preparation as cfg
from config import paths
import pandas as pd

def data_preparation():

  # Generate the data if the data source is 'minimal'  
  if cfg.data_source == 'minimal':
    sim_data_list, before_jump_list = data_generation()

  # Load the data if the data source is 'detailed'
  elif cfg.data_source == 'detailed':
    sim_data_list, before_jump_list = data_loading()

  else:
    raise ValueError('Data source not recognized.')
  
  # Concatenate the data from all simulations
  data = np.concatenate(sim_data_list)
  before_jump = np.concatenate(before_jump_list)

  # Add a column with the simulation number to data
  sim_number = np.concatenate([np.full(len(sim), i) for i, sim in enumerate(sim_data_list)])
  data = np.column_stack((data, sim_number))

  # Remove data before a jump
  data = data[~before_jump]
  print(f"Number of jumps: {np.sum(before_jump)}.")

  # Mask data where B and D are zero if specified
  if cfg.mask_zeroes:
    zeros_mask = (data[:,0] == 0) & (data[:,1] == 0)
    data = data[~zeros_mask]
    print(f"Number of zero values: {np.sum(zeros_mask)}.")

  # Drop a percentage of the data for better performance
  if cfg.drop_size > 0:
    drop_mask = np.random.uniform(0,1,len(data)) > cfg.drop_size
    data = data[drop_mask]
    print(f"Number of dropped values: {np.sum(~drop_mask)}.")
  n_dropped = np.sum(~drop_mask) if cfg.drop_size > 0 else 0

  # Calculate the number of simulations for the train, test and val sets
  n_sim_test = int(len(sim_data_list) * cfg.test_size)
  n_sim_val = int(len(sim_data_list) * cfg.val_size)
  n_sim_train = len(sim_data_list) - n_sim_test - n_sim_val

  # Make masks for the train, test and val sets
  train_mask = data[:,-1] < n_sim_train
  val_mask = (data[:,-1] >= n_sim_train) & (data[:,-1] < n_sim_train + n_sim_val)
  test_mask = data[:,-1] >= n_sim_train + n_sim_val

  # Select the data between X and y for train, val and test
  X_train, y_train = data[train_mask, :3], data[train_mask, 3:5]
  X_val, y_val = data[val_mask, :3], data[val_mask, 3:5]
  X_test, y_test = data[test_mask, :3], data[test_mask, 3:5]

  # Define headers for the data
  header_X = 'biomass (B), soil_depth (D), grazing_pressure (g)'
  header_y = 'dB_dt, dD_dt'

  # Save the processed data to individual csv files
  np.savetxt(paths.processed_data / 'X_train.csv', X_train,
             delimiter=",", header=header_X)
  np.savetxt(paths.processed_data / 'X_val.csv', X_val,
              delimiter=",", header=header_X)
  np.savetxt(paths.processed_data / 'X_test.csv', X_test,
              delimiter=",", header=header_X)
  np.savetxt(paths.processed_data / 'y_train.csv', y_train,
              delimiter=",", header=header_y)
  np.savetxt(paths.processed_data / 'y_val.csv', y_val,
              delimiter=",", header=header_y)
  np.savetxt(paths.processed_data / 'y_test.csv', y_test,
              delimiter=",", header=header_y)
  
  # Make the summary of the outputs
  dp_summary = "".join(['\n\nDATA PREPARATION:',
                        '\nn_jumps: {}'.format(np.sum(before_jump)),
                        '\nn_zeroes: {}'.format(np.sum(zeros_mask)),
                        '\nn_dropped: {}'.format(n_dropped),
                        '\ntrain_samples: {}'.format(len(X_train)),
                        '\nval_samples: {}'.format(len(X_val)),
                        '\ntest_samples: {}'.format(len(X_test))])
  
  return dp_summary

##############################################################################

def data_generation():
 # Define the physical parameters
  r, c, i, d, s = 2.1, 2.9, -0.7, 0.04, 0.4 
  Wo, a, Et, Eo, k, b, C = 5e-4, 4.02359478109, 0.021, 0.084, 0.05, 0.28, 1e-4
  alpha = np.log(Wo/C)/a

  #Define the function that computes dB/dt and dD/dt
  def dX_dt(B,D,g):
      dB_dt_step = (1-(1-i)*np.exp(-1*D/d))*(r*B*(1-B/c))-g*B/(s+B)
      dD_dt_step = Wo*np.exp(-1*a*D)-np.exp(-1*B/b)*(Et+np.exp(-1*D/k)*(Eo-Et))-C
      return dB_dt_step, dD_dt_step

  # Define the number of simulations and the number of years per simulation
  n_steps = int(cfg.n_years/cfg.dt)
  t = np.linspace(0, cfg.n_years, n_steps)

  # Initialize B, D and g
  B_jp = np.zeros((n_steps, cfg.n_sim))
  D_jp = np.zeros((n_steps, cfg.n_sim))
  g_jp = np.zeros((n_steps, cfg.n_sim))

  B_jp[0] = np.random.uniform(0, c, cfg.n_sim)
  D_jp[0] = np.random.uniform(0, alpha, cfg.n_sim)
  g_jp[0] = np.random.uniform(0, 3, cfg.n_sim)

  # Initialize dB/dt and dD/dt
  dB_dt_jp = np.zeros_like(B_jp)
  dD_dt_jp = np.zeros_like(D_jp)

  # Create a mask array to keep track of jumps
  before_jump = np.full((n_steps, cfg.n_sim), False)

  # Allow the system to evolve
  for step in range(1,n_steps):
  
      # Compute the derivatives
      steps_slopes = dX_dt(B_jp[step-1], D_jp[step-1], g_jp[step-1])
      dB_dt_jp[step-1], dD_dt_jp[step-1] = steps_slopes

      # Compute the new values, forced to be above 0 and below their theoretical max
      B_jp[step] = np.clip(B_jp[step-1] + steps_slopes[0]*cfg.dt, 0.0, c)
      D_jp[step] = np.clip(D_jp[step-1] + steps_slopes[1]*cfg.dt, 0.0, alpha)
      g_jp[step] = g_jp[step-1]

      # Add a random chance to set a new random state value
      jump_state = np.random.choice([True, False], size=len(B_jp[step]), p=[cfg.prob_new_state, 1 - cfg.prob_new_state])
      B_jp[step][jump_state] = np.random.uniform(0, c, size=len(B_jp[step][jump_state]))
      D_jp[step][jump_state] = np.random.uniform(0, alpha, size=len(D_jp[step][jump_state]))
      before_jump[step-1][jump_state] = True

      # Add a random chance to set a new random g value
      jump_g = np.random.choice([True, False], size=len(g_jp[step]), p=[cfg.prob_new_g, 1 - cfg.prob_new_g])
      g_jp[step][jump_g] = np.random.uniform(0, 3, size=len(g_jp[step][jump_g]))
      
  dB_dt_jp[-1], dD_dt_jp[-1] = dX_dt(B_jp[-1], D_jp[-1], g_jp[-1])

  # Create lists of simulations
  sim_data_list = [np.column_stack((B, D, g, dB_dt, dD_dt)) for B, D, g, dB_dt, dD_dt
                   in zip(B_jp.T, D_jp.T, g_jp.T, dB_dt_jp.T, dD_dt_jp.T)]
  before_jump_list = [bj for bj in before_jump.T]

  return sim_data_list, before_jump_list

##############################################################################

def data_loading():
  # Initialize lists to store the simulation data and before_jump arrays
  sim_data_list = []
  before_jump_list = []

  # Load the data
  data_path = paths.raw_data / cfg.data_folder

  for folder in data_path.iterdir():
    if folder.name.isdigit():
      print(f'Loading data from simulation {folder.name}')

      # Load the files
      biomass = np.loadtxt(folder / 'biomass.tss')[:,1]
      soil_depth = np.loadtxt(folder / 'soildepth.tss')[:,1]
      jumps = np.loadtxt(folder / 'statevars_jumped.tss')[:,1].astype(bool)
      grazing_pressure = np.load(folder / 'grazing.npy') * 24 * 365

      # Retrieve X from the data
      raw_X_sim = np.column_stack((biomass, soil_depth, grazing_pressure))

      # Pool the results by 26 steps
      X_sim = raw_X_sim.reshape(-1, 26, 3)
      X_sim = np.apply_along_axis(np.median, axis=1, arr=X_sim)
      jumps = jumps[::26]

      # Define the output, and remove the last step from X to match the output
      y_sim = np.column_stack((X_sim[1:,0] - X_sim[:-1,0], X_sim[1:,1] - X_sim[:-1,1]))
      X_sim = X_sim[:-1]

      # Define the mask for the data before a jump
      before_jump = np.roll(jumps, shift=-1)
      before_jump = before_jump[:-1]

      # Concatenate all the data from the simulation together
      sim_data = np.column_stack((X_sim, y_sim))

      # Append the data to the lists
      before_jump_list.append(before_jump)
      sim_data_list.append(sim_data)

  return sim_data_list, before_jump_list