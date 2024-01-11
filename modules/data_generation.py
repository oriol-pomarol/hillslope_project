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
    prob_new_state = 0.02       # probability of setting a new system state
    prob_new_g = 0.001          # probability of setting a new random g value

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
        g_jumps[step] = g_jumps[step-1]

        # Add a random chance to set a new random B value
        jump_state = np.random.choice([True, False], size=len(B_jumps[step]), p=[prob_new_state, 1 - prob_new_state])
        B_jumps[step][jump_state] = np.random.uniform(0, c, size=len(B_jumps[step][jump_state]))
        D_jumps[step][jump_state] = np.random.uniform(0, alpha, size=len(D_jumps[step][jump_state]))
        jumps_mask[step-1][jump_state] = True

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

    X_jumps_size = np.sum([len(X_jumps_filtered[i]) for i in range(n_sim)])
    print(f"Final jump data size: {X_jumps_size}")
    
    # Start the system anew for the g_increase training data

    g_init = 0.0
    B_init = c
    D_init = alpha
    dB_dt_init, dD_dt_init = dX_dt(B_init, D_init, g_init) 
    max_steps_init = int(1e5)

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

    # Generate a sequence with a quadratic g increase
    prob_disturbance = 0.001
    strength_disturbance = 0.01
    n_steps = int(1e6)
    g_inc = np.square(np.linspace(0, 2**0.5, n_steps))
    B_inc = np.ones_like(g_inc) * B_init
    D_inc = np.ones_like(g_inc) * D_init
    dB_dt_inc = np.ones_like(g_inc) * dB_dt_init
    dD_dt_inc = np.ones_like(g_inc) * dD_dt_init

    # Create a mask array to keep track of disturbances
    dist_mask = np.full(n_steps, False)

    # Allow the system to evolve
    for step in range(1, n_steps):
        B_inc[step] = np.clip(B_inc[step-1] + dB_dt_inc[step-1]*dt, 0.0, c)
        D_inc[step] = np.clip(D_inc[step-1] + dD_dt_inc[step-1]*dt, 0.0, alpha)
        dB_dt_inc[step], dD_dt_inc[step] = dX_dt(B_inc[step], D_inc[step], g_inc[step])

        # Add a random chance to disturb the system
        if np.random.uniform(0, 1) < prob_disturbance:
            B_inc[step] = np.clip(B_inc[step] + np.random.uniform(-1, 1)*strength_disturbance*c, 0.0, c)
            D_inc[step] = np.clip(D_inc[step] + np.random.uniform(-1, 1)*strength_disturbance*alpha, 0.0, alpha)
            dB_dt_inc[step], dD_dt_inc[step] = dX_dt(B_inc[step], D_inc[step], g_inc[step])
            dist_mask[step-1] = True

    # Plot the g_increase results
    years = np.linspace(0, n_steps*dt, len(g_inc))
    fig, axs = plt.subplots(3, 1, figsize = (10,7.5))

    axs[0].plot(years, D_inc, '-b', label = 'Minimal model')
    axs[0].set_ylim(0)
    axs[0].set_ylabel('soil thickness')
    axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[0].tick_params(axis="both", which="both", direction="in",
                            top=True, right=True)

    axs[1].plot(years, B_inc, '-b')
    axs[1].set_ylim(0)
    axs[1].set_ylabel('biomass')
    axs[1].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[1].tick_params(axis="both", which="both", direction="in",
                            top=True, right=True)

    axs[2].plot(years, g_inc, '-b')
    axs[2].set_ylim(0)
    axs[2].set_ylabel('grazing pressure')
    axs[2].set_xlabel('time (years)')
    axs[2].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[2].tick_params(axis="both", which="both", direction="in",
                            top=True, right=True)

    fig.patch.set_alpha(1)
    plt.savefig(f'results/train_inc.png')

    # Plot the results in a B vs D plot
    fig, axs = plt.subplots(1, 1, figsize = (10,7.5))

    axs.plot(D_inc, B_inc, '-ok', label='Minimal model')
    axs.set_ylim(0)
    axs.set_xlabel('soil thickness')
    axs.set_ylabel('biomass')
    axs.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs.tick_params(axis="both", which="both", direction="in",
                            top=True, right=True)

    fig.patch.set_alpha(1)
    plt.savefig(f'results/train_inc_BD.png')

    # Join the inc data into X and y arrays
    X_inc = np.column_stack((B_inc[~dist_mask], D_inc[~dist_mask], g_inc[~dist_mask]))
    y_inc = np.column_stack((dB_dt_inc[~dist_mask], dD_dt_inc[~dist_mask]))

    print(f"Final g_increase data size: {len(X_inc)}")

    # Report the data generation parameters
    gen_summary = "".join(['\n\n***DATA GENERATION:***',
                           '\n\nJUMPS DATA:',
                           '\nn_sim = {}'.format(n_sim),
                           '\nn_years = {}'.format(n_years),
                           '\ndt = {}'.format(dt),
                           '\nprob_new_state = {}'.format(prob_new_state),
                           '\nprob_new_g = {}'.format(prob_new_g),
                           '\nfinal_jumps_size = {}'.format(X_jumps_size),
                           '\n\nEQUILIBRIUM DATA:',
                           '\nB_init = {}'.format(B_init),
                           '\nD_init = {}'.format(D_init),
                           '\nn_steps = {}'.format(n_steps),
                           '\nprob_disturbance = {}'.format(prob_disturbance),
                           '\nstrength_disturbance = {}'.format(strength_disturbance),
                           '\nfinal_eq_size = {}'.format(len(X_inc))])

    return gen_summary, X_jumps_filtered, y_jumps_filtered, X_inc, y_inc