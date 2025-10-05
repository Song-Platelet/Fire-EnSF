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
# Font settings (Consider using LaTeX for text rendering if your paper uses LaTeX)
plt.rcParams['font.size'] = 12  # Base font size
plt.rcParams['axes.labelsize'] = 14 # Font size for x and y labels
plt.rcParams['axes.titlesize'] = 16 # Font size for the title
plt.rcParams['xtick.labelsize'] = 12 # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Font size for y-axis tick labels
plt.rcParams['legend.fontsize'] = 12 # Font size for legend


cond_alpha = lambda t: 1 - (1-eps_alpha)*t

# conditional sigma^2
# sigma2_t(0) = 0
# sigma2_t(1) = 1
# sigma(t) = t
cond_sigma_sq = lambda t: eps_beta + t * (1 - eps_beta)

# drift function of forward SDE
f = lambda t: -(1-eps_alpha) / cond_alpha(t)
# diffusion function of forward SDE
g_sq = lambda t: 1 - 2 * f(t) * cond_sigma_sq(t)
g = lambda t: np.sqrt(g_sq(t))

# generate sample with reverse SDE
def reverse_SDE(x0, score_likelihood=None, time_steps=100,
                drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,  save_path=False):
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
        sigma2_t = sigma2_fun(t)

        # Evaluate the diffusion term
        diffuse = diffuse_fun(t)

        # Evaluate the drift term
        # drift = drift_fun(t)*xt - diffuse**2 * score_eval

        # Update
        if score_likelihood is not None:
            xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) - diffuse**2 * score_likelihood(xt, t) ) \
                  + np.sqrt(dt)*diffuse*torch.randn_like(xt)
        else:
            xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) ) + np.sqrt(dt)*diffuse*torch.randn_like(xt)

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

# the simple dynamic system
forward_drift = lambda x: 2*x 

trial = 30

time_dict = {
    'ensf': np.zeros(trial),
    'enkf': np.zeros(trial)
}

for k in range(trial):
    # lorenz system
    n_dim = 2000
    SDE_sigma = 0.5

    # filtering setup
    dt = 0.05
    filtering_steps = 30

    # observation sigma
    obs_sigma = 0.1

    ####################################################################
    # EnSF setup
    # define the diffusion process
    eps_alpha = 0.96
    eps_beta = 0.04
    # ensemble size
    ensemble_size = 100

    # forward Euler step
    euler_steps = 100

    # damping function(tau(0) = 1;  tau(1) = 0;)
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

    # filtering initial ensemble
    # x_state = state_target + torch.randn(ensemble_size, n_dim, device=device)*0.5
    # x_state = torch.randn(ensemble_size, n_dim, device=device)  # pure Gaussian initial
    x = 1.15 * np.cos(angles)
    y = 1.05 * np.sin(angles)
    x_prop = torch.tensor(np.vstack((x, y)).T.flatten(), device=device) # Initial set up

    x_state = x_prop.repeat(ensemble_size, 1) + 0.1 * torch.randn(ensemble_size, n_dim, device=device)

    torch.manual_seed(114514)
    torch.cuda.empty_cache()

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
        time1 = time()

        x_prop += dt*forward_drift(x_prop)
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
        # obs: (d)
        # xt: (ensemble, d)
        # score_likelihood = lambda xt, t: -(torch.atan(xt) - obs)/obs_sigma**2 * (1./(1. + xt**2)) * g_tau(t)
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

    columns =[1] + [i for i in range(30) if i % 10 == 9]
    num_plot_a_cols = len(columns)

    if k == 0:
        # Create the subplots
        fig_a, axs_a = plt.subplots(nrows=3, ncols=num_plot_a_cols,
                                    figsize=(13, 8),
                                    sharey='row', sharex='col')

        # Define colors
        color_gt_ref = 'dodgerblue'     # Blue
        color_state_true = '#e66060' # Green
        color_x_est = '#6be64d'      # Orange

        # Define marker properties for scatter plots
        marker_size_scatter = 50  # 's' parameter for plt.scatter
        marker_alpha = 0.75

        # --- Create proxy artists for the figure-level legend ---
        legend_marker_size_line2d = 8

        gt_proxy = mlines.Line2D([], [], color=color_gt_ref, marker='o', linestyle='None',
                                markersize=legend_marker_size_line2d, label='Ground Truth',
                                markeredgecolor='k', markeredgewidth=0.5)
        obs_proxy = mlines.Line2D([], [], color=color_state_true, marker='^', linestyle='None',
                                markersize=legend_marker_size_line2d, label='Observation',
                                markeredgecolor='k', markeredgewidth=0.5)
        est_proxy = mlines.Line2D([], [], color=color_x_est, marker='s', linestyle='None',
                                markersize=legend_marker_size_line2d, label='Estimated State',
                                markeredgecolor='k', markeredgewidth=0.5)

        handles_for_legend = [gt_proxy, obs_proxy, est_proxy]

        for col_idx, step_num in enumerate(columns):
            # Ensure n_dim is an even number for reshaping
            if n_dim % 2 != 0:
                raise ValueError("n_dim must be an even number for reshaping into (y,x) pairs.")

            # Reshape data for plotting (assuming (y, x) pairs)
            # Ensure data is correctly indexed if it's a list of arrays
            current_state_data = state_save[step_num]
            current_obs_data = obs_save[step_num]
            current_est_data = est_save[step_num]

            gt_ref_plot_x = current_state_data.reshape((int(n_dim / 2), 2))[:, 0]
            gt_ref_plot_y = current_state_data.reshape((int(n_dim / 2), 2))[:, 1]

            st_true_plot_x = current_obs_data.reshape((int(n_dim / 2), 2))[:, 0]
            st_true_plot_y = current_obs_data.reshape((int(n_dim / 2), 2))[:, 1]

            x_est_plot_x = current_est_data.reshape((int(n_dim / 2), 2))[:, 0]
            x_est_plot_y = current_est_data.reshape((int(n_dim / 2), 2))[:, 1]

            # --- Row 1: Ground truth_ref vs. State target (state_true) ---
            ax1 = axs_a[0, col_idx] if num_plot_a_cols > 1 else axs_a[0]
            ax1.scatter(gt_ref_plot_x, gt_ref_plot_y, alpha=marker_alpha, color=color_gt_ref, s=marker_size_scatter, edgecolor='k', linewidth=0.5)
            ax1.scatter(st_true_plot_x, st_true_plot_y, alpha=marker_alpha, color=color_state_true, s=marker_size_scatter, marker='^', edgecolor='k', linewidth=0.5)
            ax1.set_title(f'Filtering Step {step_num+1}', fontsize=20, fontweight='medium')
            if step_num == 1:
                ax1.set_ylabel('Y-coordinate', fontsize=20)
            ax1.grid(False)

            # --- Row 2: State target (state_true) vs. Estimated state (x_est) ---
            ax2 = axs_a[1, col_idx] if num_plot_a_cols > 1 else axs_a[1]
            ax2.scatter(st_true_plot_x, st_true_plot_y, alpha=marker_alpha, color=color_state_true, s=marker_size_scatter, marker='^', edgecolor='k', linewidth=0.5)
            ax2.scatter(x_est_plot_x, x_est_plot_y, alpha=marker_alpha, color=color_x_est, s=marker_size_scatter, marker='s', edgecolor='k', linewidth=0.5)
            if step_num == 1:
                ax2.set_ylabel('Y-coordinate', fontsize=20)
            ax2.grid(False)

            # --- Row 3: Ground truth_ref vs. x_est ---
            ax3 = axs_a[2, col_idx] if num_plot_a_cols > 1 else axs_a[2]
            ax3.scatter(gt_ref_plot_x, gt_ref_plot_y, alpha=marker_alpha, color=color_gt_ref, s=marker_size_scatter, edgecolor='k', linewidth=0.5)
            ax3.scatter(x_est_plot_x, x_est_plot_y, alpha=marker_alpha, color=color_x_est, s=marker_size_scatter, marker='s', edgecolor='k', linewidth=0.5)
            ax3.set_xlabel('X-coordinate', fontsize=18)
            if step_num == 1:
                ax3.set_ylabel('Y-coordinate', fontsize=20)
            ax3.grid(False)

            # Improve tick label appearance
            for ax_row in range(3):
                current_ax = axs_a[ax_row, col_idx] if num_plot_a_cols > 1 else axs_a[ax_row]
                current_ax.tick_params(axis='both', which='major', labelsize=18)

        # --- Adjust layout and add the figure-level legend ---
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])
        fig_a.subplots_adjust(right=0.82)
        fig_a.legend(handles=handles_for_legend,
                    fontsize=20,
                    loc='center left',
                    bbox_to_anchor=(0.83, 0.5),
                    borderaxespad=0.)

        # Save the figure (optional)
        plt.savefig("num_result/num_ensf.pdf", bbox_inches='tight')
        plt.close()

    # Initialize ensemble
    x = 1.00 * np.cos(angles)
    y = 1.00 * np.sin(angles)
    state_target = torch.tensor(np.vstack((x, y)).T.flatten(), device=device)

    x_init_coords = 1.15 * np.cos(angles) # Changed variable name for clarity
    y_init_coords = 1.05 * np.sin(angles) # Changed variable name for clarity
    x_prop = torch.tensor(np.vstack((x_init_coords, y_init_coords)).T.flatten(), device=device) # Initial set up
    x_state = x_prop.repeat(ensemble_size, 1) + 0.1 * torch.randn(ensemble_size, n_dim, device=device)

    H = 0.25 * torch.eye(n_dim, device=device)

    # Information containers
    rmse_kf = [compute_rmse(state_target.reshape(-1,2).cpu().numpy(), x_prop.reshape(-1,2).cpu().numpy())]
    est_save = [] # This seems to store the same as state_save, consider if both are needed

    torch.set_default_dtype(torch.float64) # half precision
    torch.manual_seed(114514)
    torch.cuda.empty_cache()

    t = 0

    # EnKF implementation
    for i in range(filtering_steps):
        time1 = time()
        x_state += dt*forward_drift(x_state) + np.sqrt(dt)*SDE_sigma*torch.randn_like(x_state)
        state_target += dt*forward_drift(state_target) + np.sqrt(dt)*SDE_sigma*torch.randn_like(state_target)
        
        y = 0.25 * state_target + obs_sigma * torch.randn(n_dim, device=device) # observation
        
        x_mean = torch.mean(x_state, axis=0)
        e_x = x_state - x_mean

        P = e_x.T @ e_x / (ensemble_size - 1)

        y_perturbed = y.unsqueeze(0).repeat(ensemble_size, 1)
        y_perturbed += obs_sigma * torch.randn(ensemble_size, n_dim, device=device)
        
        Hx = 0.25 * x_state  # 应用观测算子 H = 0.25*I
        Hx_mean = torch.mean(Hx, axis=0)
        e_Hx = Hx - Hx_mean
        
        P_yy = e_Hx.T @ e_Hx / (ensemble_size - 1) + torch.eye(n_dim, device=device) * (obs_sigma**2)
        P_xy = e_x.T @ e_Hx / (ensemble_size - 1)

        K = P_xy @ torch.linalg.inv(P_yy)

        x_state = x_state + (y_perturbed - Hx) @ K.T

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

    if k == 0:
        # Create the subplots
        fig_a, axs_a = plt.subplots(nrows=3, ncols=num_plot_a_cols,
                                    figsize=(13, 8),
                                    sharey='row', sharex='col')

        # Define colors
        color_gt_ref = 'dodgerblue'     # Blue
        color_state_true = '#e66060' # Green
        color_x_est = '#6be64d'      # Orange

        # Define marker properties for scatter plots
        marker_size_scatter = 50  # 's' parameter for plt.scatter
        marker_alpha = 0.75

        # --- Create proxy artists for the figure-level legend ---
        legend_marker_size_line2d = 8

        gt_proxy = mlines.Line2D([], [], color=color_gt_ref, marker='o', linestyle='None',
                                markersize=legend_marker_size_line2d, label='Ground Truth',
                                markeredgecolor='k', markeredgewidth=0.5)
        obs_proxy = mlines.Line2D([], [], color=color_state_true, marker='^', linestyle='None',
                                markersize=legend_marker_size_line2d, label='Observation',
                                markeredgecolor='k', markeredgewidth=0.5)
        est_proxy = mlines.Line2D([], [], color=color_x_est, marker='s', linestyle='None',
                                markersize=legend_marker_size_line2d, label='Estimated State',
                                markeredgecolor='k', markeredgewidth=0.5)

        handles_for_legend = [gt_proxy, obs_proxy, est_proxy]

        for col_idx, step_num in enumerate(columns):
            # Ensure n_dim is an even number for reshaping
            if n_dim % 2 != 0:
                raise ValueError("n_dim must be an even number for reshaping into (y,x) pairs.")

            # Reshape data for plotting (assuming (y, x) pairs)
            # Ensure data is correctly indexed if it's a list of arrays
            current_state_data = state_save[step_num]
            current_obs_data = obs_save[step_num]
            current_est_data = est_save[step_num]

            gt_ref_plot_x = current_state_data.reshape((int(n_dim / 2), 2))[:, 0]
            gt_ref_plot_y = current_state_data.reshape((int(n_dim / 2), 2))[:, 1]

            st_true_plot_x = current_obs_data.reshape((int(n_dim / 2), 2))[:, 0]
            st_true_plot_y = current_obs_data.reshape((int(n_dim / 2), 2))[:, 1]

            x_est_plot_x = current_est_data.reshape((int(n_dim / 2), 2))[:, 0]
            x_est_plot_y = current_est_data.reshape((int(n_dim / 2), 2))[:, 1]

            # --- Row 1: Ground truth_ref vs. State target (state_true) ---
            ax1 = axs_a[0, col_idx] if num_plot_a_cols > 1 else axs_a[0]
            ax1.scatter(gt_ref_plot_x, gt_ref_plot_y, alpha=marker_alpha, color=color_gt_ref, s=marker_size_scatter, edgecolor='k', linewidth=0.5)
            ax1.scatter(st_true_plot_x, st_true_plot_y, alpha=marker_alpha, color=color_state_true, s=marker_size_scatter, marker='^', edgecolor='k', linewidth=0.5)
            ax1.set_title(f'Filtering Step {step_num+1}', fontsize=20, fontweight='medium')
            if step_num == 1:
                ax1.set_ylabel('Y-coordinate', fontsize=20)
            ax1.grid(False)

            # --- Row 2: State target (state_true) vs. Estimated state (x_est) ---
            ax2 = axs_a[1, col_idx] if num_plot_a_cols > 1 else axs_a[1]
            ax2.scatter(st_true_plot_x, st_true_plot_y, alpha=marker_alpha, color=color_state_true, s=marker_size_scatter, marker='^', edgecolor='k', linewidth=0.5)
            ax2.scatter(x_est_plot_x, x_est_plot_y, alpha=marker_alpha, color=color_x_est, s=marker_size_scatter, marker='s', edgecolor='k', linewidth=0.5)
            if step_num == 1:
                ax2.set_ylabel('Y-coordinate', fontsize=20)
            ax2.grid(False)

            # --- Row 3: Ground truth_ref vs. x_est ---
            ax3 = axs_a[2, col_idx] if num_plot_a_cols > 1 else axs_a[2]
            ax3.scatter(gt_ref_plot_x, gt_ref_plot_y, alpha=marker_alpha, color=color_gt_ref, s=marker_size_scatter, edgecolor='k', linewidth=0.5)
            ax3.scatter(x_est_plot_x, x_est_plot_y, alpha=marker_alpha, color=color_x_est, s=marker_size_scatter, marker='s', edgecolor='k', linewidth=0.5)
            ax3.set_xlabel('X-coordinate', fontsize=18)
            if step_num == 1:
                ax3.set_ylabel('Y-coordinate', fontsize=20)
            ax3.grid(False)

            # Improve tick label appearance
            for ax_row in range(3):
                current_ax = axs_a[ax_row, col_idx] if num_plot_a_cols > 1 else axs_a[ax_row]
                current_ax.tick_params(axis='both', which='major', labelsize=18)

        # --- Adjust layout and add the figure-level legend ---
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])
        fig_a.subplots_adjust(right=0.82)
        fig_a.legend(handles=handles_for_legend,
                    fontsize=20,
                    loc='center left',
                    bbox_to_anchor=(0.83, 0.5),
                    borderaxespad=0.)

        # Save the figure (optional)
        plt.savefig("num_result/num_enkf.pdf", bbox_inches='tight')
        plt.close()

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
