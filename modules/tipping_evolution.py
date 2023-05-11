import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import colors
import os

def tipping_evolution(nnetwork):

  # Define some parameters
  print('Starting tipping point evolution...')
  g_tipping = 1.92    # value of g of the tipping point
  Bo = 1.95  	        # initial value of B, ideally close to the equilibrium
  Do = 0.41           # initial value of D, ideally close to the equilibrium
  dt = 0.5 			      # time step, 7/365 in paper, 0.5 for general purpose
  n_steps = 1e4       # number of steps to run

  # Initialize B and D
  B_ev = np.ones((n_steps)) * Bo
  D_ev = np.ones((n_steps)) * Do

  

  # Allow the system to evolve
  perc_steps = n_steps//20
  for step in range(1,n_steps):
    if step%perc_steps == 0:
      print('{:.0f}% of steps completed.'.format(100*step/n_steps))
      
    # Compute the derivatives
    nn_slopes = nnetwork.predict(np.array([[B_ev[step-1], D_ev[step-1], g_tipping]]), verbose = False)

    #compute the new values
    B_ev[step] = np.clip(B_ev[step-1] + nn_slopes.squeeze()[0]*dt, 0, 7)
    D_ev[step] = np.clip(D_ev[step-1] + nn_slopes.squeeze()[1]*dt, 0, 3)
  print('Successfully completed tipping point evolution.')

  # Save the results
  saved_vars = [B_ev, D_ev, g_tipping]
  header_vars = 'B_ev, D_ev, g_tipping'
  file_path = os.path.join('results', 'tipping_evolution.csv')
  np.savetxt(file_path, np.column_stack(saved_vars), delimiter=',', header = header_vars)

  # Add a couple lines to the summary with the system evolution parameters
  t_evolution_summary = "".join(['\n\nTipping evolution {}:',
                               '\ntime_step = {}'.format(dt),
                               '\nn_steps = {}'.format(n_steps)])
  return t_evolution_summary