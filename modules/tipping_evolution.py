import numpy as np
import matplotlib.pyplot as plt
import os

def tipping_evolution(model):

  # Define some parameters
  print('Starting tipping point evolution...')
  g_tipping = 1.93    # value of g of the tipping point
  Bo = 1.95  	        # initial value of B, ideally close to the equilibrium
  Do = 0.41           # initial value of D, ideally close to the equilibrium
  dt = 0.5 			      # time step, 7/365 in paper, 0.5 for general purpose
  n_steps = 1000       # number of steps to run

  # Initialize B and D
  B_ev = np.ones((n_steps)) * Bo
  D_ev = np.ones((n_steps)) * Do

  

  # Allow the system to evolve
  perc_steps = n_steps//20
  for step in range(1,n_steps):
    if step%perc_steps == 0:
      print('{:.0f}% of steps completed.'.format(100*step/n_steps))
      
    # Compute the derivatives
    nn_slopes = model.predict(np.array([[B_ev[step-1], D_ev[step-1], g_tipping]]), verbose = False)

    #compute the new values
    B_ev[step] = np.clip(B_ev[step-1] + nn_slopes.squeeze()[0]*dt, 0, 7)
    D_ev[step] = np.clip(D_ev[step-1] + nn_slopes.squeeze()[1]*dt, 0, 3)
  print('Successfully completed tipping point evolution.')

  # Save the results
  saved_vars = [B_ev, D_ev, np.ones_like(B_ev) * g_tipping]
  header_vars = 'B_ev, D_ev, g_tipping'
  file_path = os.path.join('results', 'tipping_evolution.csv')
  np.savetxt(file_path, np.column_stack(saved_vars), delimiter=',', header = header_vars)

  # Some plot parameters
  n_sq = 18         # Number of ticks in each dimension
  B_lim = 3         # Maximum value of B in the plot
  D_lim = 0.6       # Maximum value of D in the plot

  # Make a grid of B and D values
  D_edges = np.linspace(0, D_lim, n_sq)
  B_edges = np.linspace(0, B_lim, n_sq)
  D_grid, B_grid = np.meshgrid(D_edges, B_edges)

  # Format the data to feed to the model
  g_grid =  np.ones((n_sq**2)) * g_tipping
  X_grid = np.column_stack((B_grid.flatten(), D_grid.flatten(), g_grid))

  # Use the model to predict the value of the derivatives on the grid
  Z = model.predict(X_grid).reshape((n_sq,n_sq,2))

  # Find the contour lines where the surface equals zero
  dB_dt_0 = plt.contour(D_grid, B_grid, Z[:,:,0], levels = [0.0], linewidths=0)
  dD_dt_0 = plt.contour(D_grid, B_grid, Z[:,:,1], levels = [0.0], linewidths=0)

  # Find the gradients
  grad_B, _ = np.gradient(Z[:,:,0])
  _, grad_D = np.gradient(Z[:,:,1])

  # Start the figure
  fig, ax = plt.subplots(figsize=(16,14))

  ax.xaxis.set_major_locator(plt.MaxNLocator(3))
  ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))

  for contour, gradient, color, var in [[dB_dt_0, grad_B, '#24A793', 'biomass'], [dD_dt_0, grad_D, '#C00A35', 'soil depth']]:

    # Find for what regions the equilibrium is stable
    grad_stab = gradient < 0

    # Save the lines as dashed or solid depending on the stability of the equilibrium
    dashed_lines = []
    solid_lines = [] 
    lines = contour.allsegs[0]
    for line in lines:
      line = np.flip(line, axis=1)
      indices = (np.array(line)//np.array([(B_lim+1E-5)/n_sq, (D_lim+1E-6)/n_sq])).astype(int)
      stability = grad_stab[indices[:,0], indices[:,1]]
      current_line = line[0]

      for i in range(len(line)-1):
        if (stability[i] != stability[i+1]):
          midpoint = (line[i] + line[i+1])/2
          current_line = np.vstack([current_line, midpoint])
          if stability[i]:
            solid_lines.append(current_line)
          else:
            dashed_lines.append(current_line)
          current_line = midpoint
        current_line = np.vstack([current_line, line[i+1]])

      if stability[-1]:
            solid_lines.append(current_line)
      else:
        dashed_lines.append(current_line)
    
    # Plot the lines
    for i, d_line in enumerate(dashed_lines):
      d_line = np.array(d_line)
      ax.plot(d_line[:,1], d_line[:,0], linestyle = 'dashed', linewidth=3, color = color,
              label = f'Unstable {var} nullcline' if i==0 else "")

    for i, s_line in enumerate(solid_lines):
      s_line = np.array(s_line)
      ax.plot(s_line[:,1], s_line[:,0], linestyle = 'solid', linewidth=3, color = color,
              label = f'Stable {var} nullcline' if i==0 else "")
      
  ax.plot(D_ev, B_ev, 'ko-', label = 'System evolution')
      
  # Set the axes properties and save the figure
  ax.set_ylim(0, B_lim)
  ax.set_xlim(0, D_lim)
  ax.set_ylabel('Biomass ($kg/m^2$)')
  ax.set_xlabel('Soil depth ($m$)')
  ax.legend(loc = 'best', framealpha=1)
  plt.tight_layout()
  plt.savefig(os.path.join('results','tipping_evolution.png'))

  # Add a couple lines to the summary with the system evolution parameters
  t_evolution_summary = "".join(['\n\nTipping evolution {}:',
                               '\ntime_step = {}'.format(dt),
                               '\nn_steps = {}'.format(n_steps)])
  return t_evolution_summary