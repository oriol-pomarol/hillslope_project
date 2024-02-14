from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from config import data_preparation as cfg
from config import paths

def data_preparation():

  # Generate the data if the data source is 'minimal'  
  if cfg.data_source == 'minimal':
    X, y, before_jump = data_generation()

  # Load the data if the data source is 'detailed'
  elif cfg.data_source == 'detailed':
    X, y, before_jump = data_loading()

  # Raise an error if the data source is not either 'minimal' or 'detailed'
  else:
    raise ValueError('Data source not recognized.')

  # Make a filter to remove data before a jump
  jumps_mask = ~before_jump
  X = X[jumps_mask]
  y = y[jumps_mask]
  print(f"Number of jumps: {np.sum(~jumps_mask)}.")

  # Mask data where B and D are zero if specified
  if cfg.mask_zeroes:
    zeros_mask = (X[:,0] == 0) & (X[:,1] == 0)
    X = X[~zeros_mask]
    y = y[~zeros_mask]
    print(f"Number of zero values: {np.sum(zeros_mask)}.")

  # Drop a percentage of the data for better performance
  if cfg.drop_size > 0:
    drop_mask = np.random.uniform(0,1,len(X)) > cfg.drop_size
    X = X[drop_mask]
    y = y[drop_mask]
    print(f"Number of dropped values: {np.sum(~drop_mask)}.")

  # Split the data between training, testing and validation
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=cfg.test_size,
                     shuffle=True, random_state=10)
  
  effective_val_size = cfg.val_size/(1-cfg.test_size)
  X_train, X_val, y_train, y_val =  \
    train_test_split(X_train, y_train,
                     test_size=effective_val_size,
                     shuffle=True, random_state=10)
  
  # Save the processed data to individual files
  np.save(paths.processed_data / 'X_train.npy', X_train)
  np.save(paths.processed_data / 'X_val.npy', X_val)
  np.save(paths.processed_data / 'X_test.npy', X_test)
  np.save(paths.processed_data / 'y_train.npy', y_train)
  np.save(paths.processed_data / 'y_val.npy', y_val)
  np.save(paths.processed_data / 'y_test.npy', y_test)
  
  # Make the summary of the outputs
  dp_summary = "".join(['\n\nDATA PREPARATION:',
                        '\nn_jumps: {}'.format(np.sum(~jumps_mask)),
                        '\nn_zeroes: {}'.format(np.sum(zeros_mask)),
                        '\nn_dropped: {}'.format(np.sum(~drop_mask)),
                        '\ntrain_size: {}'.format(len(X_train))])
  
  return dp_summary, [X_train, X_val, X_test, y_train, y_val, y_test]

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
  jumps = np.full((n_steps, cfg.n_sim), False)

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
      jumps[step-1][jump_state] = True

      # Add a random chance to set a new random g value
      jump_g = np.random.choice([True, False], size=len(g_jp[step]), p=[cfg.prob_new_g, 1 - cfg.prob_new_g])
      g_jp[step][jump_g] = np.random.uniform(0, 3, size=len(g_jp[step][jump_g]))
      
  dB_dt_jp[-1], dD_dt_jp[-1] = dX_dt(B_jp[-1], D_jp[-1], g_jp[-1])

  # Reshape the data
  X = np.column_stack((B_jp.reshape(-1), D_jp.reshape(-1), g_jp.reshape(-1)))
  y = np.column_stack((dB_dt_jp.reshape(-1), dD_dt_jp.reshape(-1)))
  jumps = jumps.reshape(-1)
  before_jump = np.roll(jumps, shift=-1)
  before_jump = before_jump[:-1]

  return X, y, before_jump

##############################################################################

def data_loading():
    # Initialize lists to store X and y arrays
  X_list = []
  y_list = []

  # Load the data
  data_path = paths.raw_data / cfg.data_folder

  for folder in data_path.iterdir():
    if folder.name.isdigit():
      print(f'Loading data from simulation {folder.name}')
      biomass = np.loadtxt(folder / 'biomass.tss')[:,1]
      soil_depth = np.loadtxt(folder / 'soildepth.tss')[:,1]
      jumps = np.loadtxt(folder / 'statevars_jumped.tss')[:,1].astype(bool)
      grazing_pressure = np.load(folder / 'grazing.npy') * 24 * 365
      # Retrieve X and y from the data
      raw_X_sim = np.column_stack((biomass, soil_depth, grazing_pressure))

      # Pool the results by 26 steps
      X_sim = raw_X_sim.reshape(-1, 26, 3)
      X_sim = np.apply_along_axis(np.median, axis=1, arr=X_sim)
      jumps = jumps[::26]

      # Define the output, and remove the last step
      y_sim = np.column_stack((X_sim[1:,0] - X_sim[:-1,0], X_sim[1:,1] - X_sim[:-1,1]))
      X_sim = X_sim[:-1]

      # Append the data to the lists
      X_list.append(X_sim)
      y_list.append(y_sim)

  # Concatenate all the data
  X = np.concatenate(X_list, axis=0)
  y = np.concatenate(y_list, axis=0)

  # Select data before jumps
  before_jump = np.roll(jumps, shift=-1)
  before_jump = before_jump[:-1]

  return X, y, before_jump