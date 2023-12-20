import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

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

    # Generate the data
    # Define some run parameters for the jumps training data
    n_sim = 100                 # number of simulations to run
    n_years = 20000             # maximum number of years to run, default 20000
    dt = 0.5                    # time step, 7/365 in paper, default 0.5
    n_steps = int(n_years/dt)   # number of steps to run each simulation
    prob_new_B = 0.01           # probability of setting a new random B value
    prob_new_D = 0.01           # probability of setting a new random D value
    prob_new_g = 0.01           # probability of setting a new random g value

    # Generate the time sequence
    t = np.linspace(0, n_years, n_steps)

    # Initialize B, D and g
    B_jumps = np.zeros((n_steps, n_sim))
    D_jumps = np.zeros((n_steps, n_sim))
    g_jumps = np.zeros((n_steps, n_sim))

    B_jumps[0] = np.random.uniform(0, c, n_sim)
    D_jumps[0] = np.random.uniform(0, alpha, n_sim)
    g_jumps[0] = np.random.uniform(0, 3, n_sim)

    # Initialize dB/dt and dD/dt
    dB_dt_jumps = np.zeros_like(B_jumps)
    dD_dt_jumps = np.zeros_like(D_jumps)

    # Create a mask array to keep track of jumps
    jumps_mask = np.full((n_steps, n_sim), False)

    # Allow the system to evolve
    for step in range(1,n_steps):
    
        # Compute the derivatives
        steps_slopes = dX_dt(B_jumps[step-1], D_jumps[step-1], g_jumps[step-1])
        dB_dt_jumps[step-1], dD_dt_jumps[step-1] = steps_slopes

        # Compute the new values, forced to be above 0 and below their theoretical max
        B_jumps[step] = np.clip(B_jumps[step-1] + steps_slopes[0]*dt, 0.0, c)
        D_jumps[step] = np.clip(D_jumps[step-1] + steps_slopes[1]*dt, 0.0, alpha)

        # Add a random chance to set a new random B value
        jump_B = np.random.choice([True, False], size=len(B_jumps[step]), p=[prob_new_B, 1 - prob_new_B])
        B_jumps[step][jump_B] = np.random.uniform(0, c, size=len(B_jumps[step][jump_B]))
        jumps_mask[step-1][jump_B] = True

        # Add a random chance to set a new random D value
        jump_D = np.random.choice([True, False], size=len(D_jumps[step]), p=[prob_new_D, 1 - prob_new_D])
        D_jumps[step][jump_D] = np.random.uniform(0, alpha, size=len(D_jumps[step][jump_D]))
        jumps_mask[step-1][jump_D] = True

        # Add a random chance to set a new random g value
        jump_g = np.random.choice([True, False], size=len(g_jumps[step]), p=[prob_new_g, 1 - prob_new_g])
        g_jumps[step][jump_g] = np.random.uniform(0, 3, size=len(g_jumps[step][jump_g]))
        
    dB_dt_jumps[-1], dD_dt_jumps[-1] = dX_dt(B_jumps[-1], D_jumps[-1], g_jumps[-1])

    # Plot D(t), B(t) and g(t) for the first simulation
    fig, axs = plt.subplots(3, 1, figsize = (10,7.5))

    axs[0].plot(t, D_jumps[:,0], '-b', label = 'Minimal model')
    axs[0].set_ylim(0)
    axs[0].set_ylabel('soil thickness')
    axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[0].tick_params(axis="both", which="both", direction="in", 
                            top=True, right=True)

    axs[1].plot(t, B_jumps[:,0], '-b')
    axs[1].set_ylim(0)
    axs[1].set_ylabel('biomass')
    axs[1].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[1].tick_params(axis="both", which="both", direction="in", 
                            top=True, right=True)

    axs[2].plot(t, g_jumps[:,0], '-b')
    axs[2].set_ylim(0)
    axs[2].set_ylabel('grazing pressure')
    axs[2].set_xlabel('time (years)')
    axs[2].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[2].tick_params(axis="both", which="both", direction="in", 
                            top=True, right=True)

    fig.patch.set_alpha(1)
    plt.setp(axs, xlim=(0, n_years))
    plt.savefig(f'results/train_jumps_sim_0.png')

    # Mask data where B and D are zero
    zeros_mask = (B_jumps == 0.0) & (D_jumps == 0.0)
    train_mask_jumps = ~(jumps_mask | zeros_mask)

    print(f"{np.sum(zeros_mask)} zero values found.")

    # Change the data into a list of arrays corresponding to each simulation
    B_jumps = [np.squeeze(arr) for arr in np.split(B_jumps, n_sim, axis=1)]
    D_jumps = [np.squeeze(arr) for arr in np.split(D_jumps, n_sim, axis=1)]
    g_jumps = [np.squeeze(arr) for arr in np.split(g_jumps, n_sim, axis=1)]
    dB_dt_jumps = [np.squeeze(arr) for arr in np.split(dB_dt_jumps, n_sim, axis=1)]
    dD_dt_jumps = [np.squeeze(arr) for arr in np.split(dD_dt_jumps, n_sim, axis=1)]
    train_mask_jumps = [np.squeeze(arr) for arr in np.split(train_mask_jumps, n_sim, axis=1)]

    # Join the data into X and y arrays
    X_jumps = [np.column_stack((B_jumps[i], D_jumps[i], g_jumps[i])) for i in range(n_sim)]
    y_jumps = [np.column_stack((dB_dt_jumps[i], dD_dt_jumps[i])) for i in range(n_sim)]

    # Filter the jumps data
    X_jumps_filtered = [np.compress(train_mask_jumps[i], X_jumps[i], axis=0) for i in range(n_sim)]
    y_jumps_filtered = [np.compress(train_mask_jumps[i], y_jumps[i], axis=0) for i in range(n_sim)]

    print(f"Final jump data size: {np.sum([len(X_jumps_filtered[i]) for i in range(n_sim)])}")
    
    # Start the system anew for the linear training data
    g_init = 0.0
    B_init = c
    D_init = alpha
    dB_dt_init, dD_dt_init = dX_dt(B_init, D_init, g_init) 
    max_steps_init = int(1e4)

    # Allow the system to evolve until it reaches equilibrium (at g=0)
    for step in range(0, max_steps_init):

        # Check if equilibrium is reached
        if (abs(dB_dt_init) < 1e-8 and abs(dD_dt_init) < 1e-9):
            break

        # Compute the new values, forced between 0 and their theoretical max
        B_init = np.clip(B_init + dB_dt_init*dt, 0.0, c)
        D_init = np.clip(D_init + dD_dt_init*dt, 0.0, alpha)
        dB_dt_init, dD_dt_init = dX_dt(B_init, D_init, g_init)

    print(f"Initial values: B = {B_init}, D = {D_init} after {step+1} steps.")
    print(f"|dB_dt| = {abs(dB_dt_init)}, |dD_dt| = {abs(dD_dt_init)}.")

    # Generate a sequence with a linear g increase
    n_steps = int(1e6)
    g_lin = np.linspace(0, 2, n_steps)
    B_lin = np.ones_like(g_lin) * B_init
    D_lin = np.ones_like(g_lin) * D_init
    dB_dt_lin = np.ones_like(g_lin) * dB_dt_init
    dD_dt_lin = np.ones_like(g_lin) * dD_dt_init

    # Allow the system to evolve
    for step in range(1, n_steps):
        B_lin[step] = np.clip(B_lin[step-1] + dB_dt_lin[step-1]*dt, 0.0, c)
        D_lin[step] = np.clip(D_lin[step-1] + dD_dt_lin[step-1]*dt, 0.0, alpha)
        dB_dt_lin[step], dD_dt_lin[step] = dX_dt(B_lin[step], D_lin[step], g_lin[step])

    # Plot the linear g increase results
    years = np.linspace(0, n_steps*dt, len(g_lin))
    fig, axs = plt.subplots(3, 1, figsize = (10,7.5))

    axs[0].plot(years, D_lin, '-b', label = 'Minimal model')
    axs[0].set_ylim(0)
    axs[0].set_ylabel('soil thickness')
    axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[0].tick_params(axis="both", which="both", direction="in",
                            top=True, right=True)

    axs[1].plot(years, B_lin, '-b')
    axs[1].set_ylim(0)
    axs[1].set_ylabel('biomass')
    axs[1].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[1].tick_params(axis="both", which="both", direction="in",
                            top=True, right=True)

    axs[2].plot(years, g_lin, '-b')
    axs[2].set_ylim(0)
    axs[2].set_ylabel('grazing pressure')
    axs[2].set_xlabel('time (years)')
    axs[2].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[2].tick_params(axis="both", which="both", direction="in",
                            top=True, right=True)

    fig.patch.set_alpha(1)
    plt.savefig(f'results/train_lin.png')

    # Plot the results in a B vs D plot
    fig, axs = plt.subplots(1, 1, figsize = (10,7.5))

    axs.plot(D_lin, B_lin, '-ok', label='Minimal model')
    axs.set_ylim(0)
    axs.set_xlabel('soil thickness')
    axs.set_ylabel('biomass')
    axs.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs.tick_params(axis="both", which="both", direction="in",
                            top=True, right=True)

    fig.patch.set_alpha(1)
    plt.savefig(f'results/train_lin_BD.png')

    # Join the lin data into X and y arrays
    X_lin = np.column_stack((B_lin, D_lin, g_lin))
    y_lin = np.column_stack((dB_dt_lin, dD_dt_lin))

    print(f"Final linear data size: {len(X_lin)}")

    return(X_jumps_filtered, y_jumps_filtered, X_lin, y_lin)