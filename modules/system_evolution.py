import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import colors
import os

def system_evolution(nnetwork, rforest, X_ev, iter_count=None):

  # Import the validation data
  if isinstance(X_ev, str):
    with open(os.path.join('data', X_ev + '.pkl'), 'rb') as f:
        B_det, D_det, g_ev, B_std, D_std = pickle.load(f)
  else:
    B_det, D_det, g_ev = X_ev

  # Define the physical parameters
  r, c, i, d, s = 2.1, 2.9, -0.7, 0.04, 0.4 
  Wo, a, Et, Eo, k, b, C = 5e-4, 4.02359478109, 0.021, 0.084, 0.05, 0.28, 1e-4
  alpha = np.log(Wo/C)/a

  # Define the function that computes dB/dt and dD/dt
  def dX_dt(B,D,g):
    dB_dt_step = (1-(1-i)*np.exp(-1*D/d))*(r*B*(1-B/c))-g*B/(s+B)
    dD_dt_step = Wo*np.exp(-1*a*D)-np.exp(-1*B/b)*(Et+np.exp(-1*D/k)*(Eo-Et))-C
    return dB_dt_step*(B!=0), dD_dt_step*(D!=0)

  # Define some run parameters
  print(f'Sim {iter_count}: Starting system evolution...')
  Bo = B_det[0]	  # initial value of B
  Do = D_det[0]   # initial value of D
  dt = 0.5 			  # time step, 7/365 in paper, 0.1 for stability in results
  n_years = 20000   # maximum number of years to run, 20000 in paper

    

  # Compute the number of steps, t sequence, and when to show the percentages
  n_years = min(n_years, dt*len(B_det))
  n_steps = int(n_years/dt)
  t = np.linspace(0, n_years, n_steps)
  perc_steps = n_steps//20

  # Initialize B and D
  B_for = np.ones((n_steps)) * Bo
  D_for = np.ones((n_steps)) * Do

  B_nn = np.ones((n_steps)) * Bo
  D_nn = np.ones((n_steps)) * Do

  # Allow the system to evolve
  for step in range(1,n_steps):
    if step%perc_steps == 0:
      print('Sim {}: {:.0f}% of steps completed.'.format(iter_count, 100*step/n_steps))
      
    # Compute the derivatives
    nn_slopes = nnetwork.predict(np.array([[B_nn[step-1], D_nn[step-1], g_ev[step-1]]]), verbose = False)
    for_slopes = rforest.predict(np.array([[B_for[step-1], D_for[step-1], g_ev[step-1]]]))

    #compute the new values, forced to be within the physically possible results
    B_for[step] = np.clip(B_for[step-1] + for_slopes.squeeze()[0]*dt, 0.01, c)
    D_for[step] = np.clip(D_for[step-1] + for_slopes.squeeze()[1]*dt, 0.01, alpha)
    B_nn[step] = np.clip(B_nn[step-1] + nn_slopes.squeeze()[0]*dt, 0.01, c)
    D_nn[step] = np.clip(D_nn[step-1] + nn_slopes.squeeze()[1]*dt, 0.01, alpha)

  # Plot D(t), B(t) and g(t)
  fig, axs = plt.subplots(3, 1, figsize = (10,7.5))

  if isinstance(X_ev, str):
    axs[1].fill_between(t, (B_det-B_std)[:n_steps], (B_det+B_std)[:n_steps],
                  color='lightskyblue', alpha = 0.3, linewidth=0)
    axs[0].fill_between(t, (D_det-D_std)[:n_steps], (D_det+D_std)[:n_steps], 
                    color='lightskyblue', alpha = 0.3, linewidth=0)

  axs[0].plot(t, D_det[:n_steps], '-b', label = 'Detailed model')
  axs[0].plot(t, D_nn, '-r', label = 'Neural network')
  axs[0].plot(t, D_for, '-g', label = 'Random forests')
  axs[0].set_ylim(0)
  axs[0].set_ylabel('soil thickness')
  axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
  axs[0].tick_params(axis="both", which="both", direction="in", 
                         top=True, right=True)

  axs[1].plot(t, B_det[:n_steps], '-b')
  axs[1].plot(t, B_nn, '-r')
  axs[1].plot(t, B_for, '-g')
  axs[1].set_ylim(0)
  axs[1].set_ylabel('biomass')
  axs[1].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
  axs[1].tick_params(axis="both", which="both", direction="in", 
                         top=True, right=True)

  axs[2].plot(t, g_ev[:n_steps], '-b')
  axs[2].set_ylim(0)
  axs[2].set_ylabel('grazing pressure')
  axs[2].set_xlabel('time (years)')
  axs[2].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
  axs[2].tick_params(axis="both", which="both", direction="in", 
                         top=True, right=True)

  fig.patch.set_alpha(1)
  plt.setp(axs, xlim=(0, n_years))
  if isinstance(X_ev, str):
    plt.savefig(f'results/system_evolution_{X_ev}.png')
  else:
    plt.savefig(f'results/system_evolution_train_{iter_count}.png')
  print(f'Sim {iter_count}: Successfully completed system evolution.')

  # Save the results
  saved_vars = [B_det[:n_steps], B_for, B_nn, D_det[:n_steps], D_for, D_nn, g_ev[:n_steps], t]
  header_vars = 'B_det,B_steps,B_for,B_nn,D_det,D_steps,D_for,D_nn,g,t'

  if isinstance(X_ev, str):
    saved_vars.extend([B_std[:n_steps], D_std[:n_steps]])
    header_vars += ',B_std,D_std'
    file_path = f'results/system_evolution_{X_ev}.csv'
  else:
    file_path = f'results/system_evolution_train_{iter_count}.csv'

  np.savetxt(file_path, np.column_stack(saved_vars), delimiter=',', header = header_vars)

  r_for = np.corrcoef(B_det[:n_steps], B_for)[0, 1], np.corrcoef(D_det[:n_steps], D_for)[0, 1]
  r_nn = np.corrcoef(B_det[:n_steps], B_nn)[0, 1], np.corrcoef(D_det[:n_steps], D_nn)[0, 1]

  # Add a couple lines to the summary with the system evolution parameters
  evolution_summary = "".join(['\n\nSimulation {}:'.format(iter_count),
                               '\ntime_step = {}'.format(dt),
                               '\nn_years = {}'.format(n_years),
                               '\npearson_corr_for = {}'.format(r_for),
                               '\npearson_corr_nn = {}'.format(r_nn)])
  return evolution_summary