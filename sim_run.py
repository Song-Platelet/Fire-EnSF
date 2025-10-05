import os

import torch
import numpy as np
import pandas as pd

from time import time

from util import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Apply a style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12  # Base font size
plt.rcParams['axes.labelsize'] = 14 # Font size for x and y labels
plt.rcParams['axes.titlesize'] = 16 # Font size for the title
plt.rcParams['xtick.labelsize'] = 12 # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Font size for y-axis tick labels
plt.rcParams['legend.fontsize'] = 12 # Font size for legend

# conditional alpha^2
cond_alpha = lambda t: 1 - (1-eps_alpha)*t

# conditional beta^2
cond_beta_sq = lambda t: eps_beta + t * (1 - eps_beta)

# drift function of forward SDE
f = lambda t: -(1-eps_alpha) / cond_alpha(t)
# diffusion function of forward SDE
g_sq = lambda t: 1 - 2 * f(t) * cond_beta_sq(t)
g = lambda t: np.sqrt(g_sq(t))

# generate sample with reverse SDE
def reverse_SDE(x0, score_likelihood=None, time_steps=100,
                drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, beta2_fun=cond_beta_sq,  save_path=False):
    # x_T: sample from standard Gaussian
    # x_0: target distribution to sample from

    # reverse SDE sampling process
    # N1 = x_T.shape[0]
    # N2 = x0.shape[0]
    # d = x_T.shape[1]

    # Generate the time mesh
    dt = 1.0/time_steps

    # Initialization
    xt = torch.randn(ensemble_size, n_dim, device=device)
    t = 1.0

    # define storage
    if save_path:
        path_all = [xt]
        t_vec = [t]

    # forward Euler sampling
    for i in range(time_steps):
        # prior score evaluation
        alpha_t = alpha_fun(t)
        beta2_t = beta2_fun(t)

        # Evaluate the diffusion term
        diffuse = diffuse_fun(t)

        # Evaluate the drift term
        # drift = drift_fun(t)*xt - diffuse**2 * score_eval

        # Update
        if score_likelihood is not None:
            xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/beta2_t) - diffuse**2 * score_likelihood(xt, t) ) \
                  + np.sqrt(dt)*diffuse*torch.randn_like(xt)
        else:
            xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/beta2_t) ) + np.sqrt(dt)*diffuse*torch.randn_like(xt)

        # Store the state in the path
        if save_path:
            path_all.append(xt)
            t_vec.append(t)

        # update time
        t = t - dt

    if save_path:
        return path_all, t_vec
    else:
        return xt

# Simple dynamic system
forward_drift = lambda x: 2*x 

# number of trials
trial = 30

# Time record for each method
time_dict = {
    'ensf': np.zeros(trial),
    'enkf': np.zeros(trial)
}

for k in range(trial):
    # System step
    n_dim = 2000 
    SDE_sigma = 0.5

    # filtering setup
    dt = 0.05
    filtering_steps = 30

    # observation sigma
    obs_sigma = 0.1

    ####################################################################
    # EnSF setup
    # define the diffusion process (fine-Tuned)
    eps_alpha = 0.96
    eps_beta = 0.03
    # ensemble size
    ensemble_size = 100

    # forward Euler step
    euler_steps = 100

    # damping function
    g_tau = lambda t: 1-t

    # computation setting
    torch.set_default_dtype(torch.float64) # half precision
    device = torch.device('cuda')
    
    ####################################################################
    ####################################################################
    # initial state
    angles = np.linspace(-2 * np.pi, 2 * np.pi, int(n_dim/2), endpoint=False)
    x = 1.00 * np.cos(angles)
    y = 1.00 * np.sin(angles)
    state_target = torch.tensor(np.vstack((x, y)).T.flatten(), device=device)

    # Initial setup
    x = 1.15 * np.cos(angles)
    y = 1.05 * np.sin(angles)
    x_prop = torch.tensor(np.vstack((x, y)).T.flatten(), device=device) # use for forecasting w/o DA method
    x_state = x_prop.repeat(ensemble_size, 1) + 0.1 * torch.randn(ensemble_size, n_dim, device=device) # ground truth

    torch.manual_seed(114514)
    torch.cuda.empty_cache()

    # time record
    t = 0

    # info containers
    rmse_all = [compute_rmse(state_target.reshape(-1,2).cpu().numpy(), x_prop.reshape(-1,2).cpu().numpy())]
    rmse_ori = [compute_rmse(state_target.reshape(-1,2).cpu().numpy(), x_prop.reshape(-1,2).cpu().numpy())]
    state_save = []
    obs_save = []
    est_save = []

    obs_kf = []
    state_kf = []

    # filtering cycles
    for i in range(filtering_steps):
        x_prop += dt*forward_drift(x_prop)
      
        # record the total time of the process of predict and update step
        time1 = time()
        
        # prediction step ############################################
        # state forward in time
        x_state += dt*forward_drift(x_state) + np.sqrt(dt)*SDE_sigma*torch.randn_like(x_state)
        
        # ensemble prediction (Ground Truth)
        state_target += dt*forward_drift(state_target) + np.sqrt(dt)*SDE_sigma*torch.randn_like(state_target)

        # update step ################################################
        # get observation
        # obs = torch.atan(state_target) + torch.randn_like(state_target)*obs_sigma
        obs = 0.25 * state_target + torch.randn_like(state_target) * obs_sigma

        # define likelihood score
        score_likelihood = lambda xt, t: -(0.25*xt - obs) / obs_sigma**2 * g_tau(t) * 0.25

        # generate posterior sample
        x_state = reverse_SDE(x0=x_state, score_likelihood=score_likelihood, time_steps=euler_steps)
      
        # get state estimates
        x_est = torch.mean(x_state, dim=0)

        t += time() - time1

        # get rmse
        rmse_ensf = compute_rmse(x_est.reshape(-1,2).cpu().numpy(), state_target.reshape(-1,2).cpu().numpy())#torch.sqrt(torch.mean((x_est - state_target)**2)).item()
        rmse_o = compute_rmse(x_prop.reshape(-1,2).cpu().numpy(), state_target.reshape(-1,2).cpu().numpy())

        if x_state.device.type == 'cuda':
            torch.cuda.current_stream().synchronize() #Wait for all kernels in all streams on a CUDA device to complete.
        if rmse_ensf > 1000:
            print('diverge!')
            break
        
        # save information
        rmse_ori.append(rmse_o)
        rmse_all.append(rmse_ensf)

        obs_kf.append(obs)
        state_kf.append(state_target.clone())

        state_save.append(state_target.clone())
        obs_save.append(obs)
        est_save.append(x_est)

    # save results
    state_save = torch.stack(state_save, dim=0).cpu().numpy()

    obs_save = torch.stack(obs_save, dim=0).cpu().numpy()
    est_save = torch.stack(est_save, dim=0).cpu().numpy()

    time_dict['ensf'][k] = t



    ####################################################################
    # EnKF setup
    # Initialize ensemble
    x = 1.00 * np.cos(angles)
    y = 1.00 * np.sin(angles)
    state_target = torch.tensor(np.vstack((x, y)).T.flatten(), device=device)

    # Initial setup
    x_init_coords = 1.15 * np.cos(angles) # Changed variable name for clarity
    y_init_coords = 1.05 * np.sin(angles) # Changed variable name for clarity
    x_prop = torch.tensor(np.vstack((x_init_coords, y_init_coords)).T.flatten(), device=device)
    x_state = x_prop.repeat(ensemble_size, 1) + 0.1 * torch.randn(ensemble_size, n_dim, device=device)

    # Information containers for EnKF
    rmse_kf = [compute_rmse(state_target.reshape(-1,2).cpu().numpy(), x_prop.reshape(-1,2).cpu().numpy())]
    est_save = [] # This seems to store the same as state_save, consider if both are needed

    torch.set_default_dtype(torch.float64) # half precision
    torch.manual_seed(114514)
    torch.cuda.empty_cache()

    t = 0

    # EnKF implementation
    for i in range(filtering_steps):
        time1 = time()
        # predict step
        x_state += dt*forward_drift(x_state) + np.sqrt(dt)*SDE_sigma*torch.randn_like(x_state)
        state_target += dt*forward_drift(state_target) + np.sqrt(dt)*SDE_sigma*torch.randn_like(state_target)
        
        # create an observation y from the true state with added noise
        y = 0.25 * state_target + obs_sigma * torch.randn(n_dim, device=device)
        
        # calculate the ensemble perturbation matrix
        x_mean = torch.mean(x_state, axis=0)
        e_x = x_state - x_mean

        # estimate the forecast error covariance matrix from the ensemble
        P = e_x.T @ e_x / (ensemble_size - 1)
        
        # create perturbed observations
        y_perturbed = y.unsqueeze(0).repeat(ensemble_size, 1)
        y_perturbed += obs_sigma * torch.randn(ensemble_size, n_dim, device=device)
        
        # estimate the mean of the predicted observations
        Hx = 0.25 * x_state 
        Hx_mean = torch.mean(Hx, axis=0)
        e_Hx = Hx - Hx_mean # predicted observation perturbation matrix
        
        # innovation covariance matrix
        P_yy = e_Hx.T @ e_Hx / (ensemble_size - 1) + torch.eye(n_dim, device=device) * (obs_sigma**2)
        # cross-covariance matrix
        P_xy = e_x.T @ e_Hx / (ensemble_size - 1)

        K = P_xy @ torch.linalg.inv(P_yy) # Kalman Gain

        # update each ensemble member using the Kalman gain
        x_state = x_state + (y_perturbed - Hx) @ K.T

        # final state estimate
        x_est = torch.mean(x_state, axis=0)

        t += time() - time1
        if x_state.device.type == 'cuda':
            torch.cuda.current_stream().synchronize()

        rmse = compute_rmse(x_est.reshape(-1,2).cpu().numpy(), state_target.reshape(-1,2).cpu().numpy())
        rmse_kf.append(rmse.item())
        est_save.append(x_est)

    # Convert results to numpy arrays
    rmse_kf = np.array(rmse_kf)
    est_save = torch.stack(est_save, dim=0).cpu().numpy()
    time_dict['enkf'][k] = t

####################################################################
# Plot of RMSE Comparison
# Line and marker settings
line_width = 1
marker_size = 6

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(8, 5)) # Adjust figsize as needed for your paper's column width

# Plotting the data
ax.plot(rmse_all, label='EnSF', linewidth=line_width, color= '#3850a1', marker='o', markersize=marker_size) # Added markers every 5 points
ax.plot(rmse_kf, label='EnKF', linewidth=line_width, color= '#e8252d', marker='^', markersize=marker_size)
ax.plot(rmse_ori, label='Original', linewidth=line_width, color= '#FF6BA7', marker='s', markersize=marker_size) # Added markers every 5 points

# Setting labels and title
ax.set_xlabel("Filtering Step", fontsize=plt.rcParams['axes.labelsize'])
ax.set_ylabel("RMSE (Root Mean Square Error) \n Haversine Distance (km)", fontsize=plt.rcParams['axes.labelsize']) # Be more descriptive with y-label

# Legend
legend = ax.legend(frameon=True, loc='best') # 'best' tries to find the least obstructive location
legend.get_frame().set_edgecolor('gray') # Add a light border to the legend

# Grid
ax.grid(True, linestyle='--', alpha=0.5, axis = 'y') # Customize grid style
ax.grid(False, axis = 'x') # Customize grid style

# Spine visibility (optional: remove top and right spines for a cleaner look)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Adjust layout to prevent labels from being cut off
plt.tight_layout()
plt.savefig("num_result/num_com.pdf", bbox_inches='tight')
plt.close()

data = pd.DataFrame(time_dict)

with open('time_com_result.txt', 'w') as f:
    f.write(data.describe().to_string())
