import os
import numpy as np
from sklearn.model_selection import train_test_split

def detailed_data_formatting():

  # Initialize lists to store X and y arrays
  X_list = []
  y_list = []

  # Load the data
  for folder in os.listdir('data/detailed_jp'):
    if folder.isdigit():
      print('Loading data from simulation {}'.format(folder))
      path = os.path.join('data', 'detailed_jp', folder)
      biomass = np.loadtxt(os.path.join(path, 'biomass.tss'))[:,1]
      soil_depth = np.loadtxt(os.path.join(path, 'soildepth.tss'))[:,1]
      jumps = np.loadtxt(os.path.join(path, 'statevars_jumped.tss'))[:,1]
      grazing_pressure = np.load(os.path.join(path, 'grazing.npy'))*24*365
      
      # Retrieve X and y from the data
      raw_X_sim = np.column_stack((biomass, soil_depth, grazing_pressure))

      # Pool the results by taking the median every 26 steps
      X_sim = raw_X_sim.reshape(-1, 26, 3)
      X_sim = np.apply_along_axis(np.median, axis=1, arr=X_sim)

      # Define the output
      y_sim = np.column_stack((X[1:,0] - X[:-1,0], X[1:,1] - X[:-1,1]))

      # Find jumps every 26 steps
      jumps = jumps.astype(bool)
      jumps = jumps[::26]

      # Make a filter to remove data before a jump
      before_jump = np.roll(jumps, shift=-1)
      jumps_filter = ~before_jump

      # Remove the last value (no matching y)
      jumps_filter[-1] = False

      # Filter the data
      X_sim = X_sim[jumps_filter]
      y_sim = y_sim[jumps_filter[:-1]]

      # Append the data to the lists
      X_list.append(X_sim)
      y_list.append(y_sim)

  # Concatenate all the data
  X = np.concatenate(X_list, axis=0)
  y = np.concatenate(y_list, axis=0)

  # Drop a percentage of the data for better performance
  drop_size = 0.9
  drop_mask = np.random.uniform(0,1,len(X)) >= drop_size
  X = X[drop_mask]
  y = y[drop_mask]


  # Split the data between training, testing and validation
  test_size = 0.2
  val_size = 0.1
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                      shuffle=True, random_state=10)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    test_size=val_size/(1-test_size),
                                                    shuffle=True, random_state=10)
  
  
  # Add the data characteristics to the summary
  data_summary = "".join(['\n\n***DATA FORMATTING***',
                          '\ndata_type = detailed',
                          '\ntest_size = {}'.format(test_size),
                          '\nval_size = {}'.format(val_size),
                          '\nfinal_train_size = {}'.format(X_train.shape[0])])

  return data_summary, [X_train, X_val, y_train, y_val], [X_test, y_test], [None]*3

def minimal_data_formatting(jp_eq_data):

  # Unpack the data
  X_jp, y_jp, X_eq, y_eq = jp_eq_data

  # Set the weights for the equilibrium and jumps data
  w_eq = 0       # between 0 and 1

  # Split the test data
  test_size = 0.2

  # Split the equilibrium data between testing and the rest
  X_eq_train, X_eq_test, y_eq_train, y_eq_test = \
    train_test_split(X_eq, y_eq, test_size=test_size,
                     shuffle=True, random_state=10)
  
  # Split the jumps data between testing and the rest and concatenate
  n_sim_test = int(len(X_jp)*test_size)
  X_jp_train = np.concatenate(X_jp[:-n_sim_test])
  y_jp_train = np.concatenate(y_jp[:-n_sim_test])
  X_jp_test = np.concatenate(X_jp[-n_sim_test:])
  y_jp_test = np.concatenate(y_jp[-n_sim_test:])

  # Split all the data between training and validation
  val_size = 0.1
  X_eq_train, X_eq_val, y_eq_train, y_eq_val = \
    train_test_split(X_eq_train, y_eq_train,
                      test_size=val_size/(1-test_size),
                      shuffle=True, random_state=10)
  X_jp_train, X_jp_val, y_jp_train, y_jp_val = \
    train_test_split(X_jp_train, y_jp_train,
                      test_size=val_size/(1-test_size),
                      shuffle=True, random_state=10)
  
  if w_eq < 1 and w_eq > 0:
    # Join the equilibrium and jumps data
    X_train = np.concatenate((X_eq_train, X_jp_train))
    y_train = np.concatenate((y_eq_train, y_jp_train))
    X_test = np.concatenate((X_eq_test, X_jp_test))
    y_test = np.concatenate((y_eq_test, y_jp_test))
    X_val = np.concatenate((X_eq_val, X_jp_val))
    y_val = np.concatenate((y_eq_val, y_jp_val))

    # Generate the weights for the equilibrium and jumps data
    len_eq_val = len(X_eq_val)
    len_eq_train = len(X_eq_train)
    length_ratio = len(X_jp_train)/len_eq_train
    w_train = np.concatenate((w_eq*np.ones(len(X_eq_train))*length_ratio,
                             (1-w_eq)*np.ones(len(X_jp_train))))

  elif w_eq == 0:
    # Use only the jumps data
    X_train = X_jp_train
    y_train = y_jp_train
    X_test = X_jp_test
    y_test = y_jp_test
    X_val = X_jp_val
    y_val = y_jp_val

  elif w_eq == 1:
    # Use only the eq data
    X_train = X_eq_train
    y_train = y_eq_train
    X_test = X_eq_test
    y_test = y_eq_test
    X_val = X_eq_val
    y_val = y_eq_val
  
  # Drop a percentage of the training data for better performance
  drop_size = 0

  if drop_size > 0:
    drop_mask, len_eq_train, _ = subset_mask_stratified(1-drop_size, len(X_train), len(X_eq_train))
    X_train = X_train[drop_mask]
    y_train = y_train[drop_mask]
    if w_train is not None:
      w_train = w_train[drop_mask] 
    drop_mask, _, _ = subset_mask_stratified(1-drop_size, len(X_val), len(X_eq_val))
    X_val = X_val[drop_mask]
    y_val = y_val[drop_mask]

  # Calculate the final training set size
  final_train_size = X_train.shape[0]
  print('Dropped {:.1f}% of the training data.'.format(100*drop_size))
  print('Final training set size: {}'.format(final_train_size))
  
  # Add the data characteristics to the summary
  data_summary = "".join(['\n\n***DATA FORMATTING***',
                          '\nw_eq = {}'.format(w_eq),
                          '\ntest_size = {}'.format(test_size),
                          '\nval_size = {}'.format(val_size),
                          '\ndrop_size = {}'.format(drop_size),
                          '\nfinal_train_size = {}'.format(final_train_size)])
  add_train_vars = [w_train, len_eq_train, len_eq_val] if (w_eq<1 and w_eq>0) else [None]*3
  return data_summary, [X_train, X_val, y_train, y_val], [X_test, y_test], \
    add_train_vars

def subset_mask_stratified(subset_factor, total_len, first_group_len):
  if first_group_len is None:
    first_group_len = total_len
  first_subset_len = int(subset_factor*first_group_len)
  second_subset_len = int(subset_factor*total_len) - first_subset_len
  subset_mask_first = np.concatenate((np.ones(first_subset_len), np.zeros(first_group_len - first_subset_len)))
  subset_mask_second = np.concatenate((np.ones(second_subset_len), np.zeros(total_len - first_group_len - second_subset_len)))
  subset_mask_first = np.random.permutation(subset_mask_first)
  subset_mask_second = np.random.permutation(subset_mask_second)
  subset_mask = np.concatenate((subset_mask_first, subset_mask_second))
  subset_mask = subset_mask.astype(bool)
  return subset_mask, first_subset_len, second_subset_len