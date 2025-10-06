import os
import sqlite3
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle

import geopandas as gpd
from shapely.ops import unary_union, polygonize
from shapely.geometry import shape, Point, Polygon, MultiPoint, MultiPolygon, LineString, MultiLineString

import torch
from util import *

def create_database_and_table(db_name="my_database.db"):
    """
    Connects to an SQLite database (creates it if it doesn't exist),
    and creates a 'users' table.
    """
    try:
        # Connect to SQLite database. If the database does not exist, it will be created.
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create a table named 'users' if it doesn't already exist.
        # It has three columns: id (INTEGER PRIMARY KEY), name (TEXT), and age (INTEGER).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                fireid INTEGER NOT NULL,
                eps_alpha double NOT NULL,
                eps_beta double NOT NULL,
                rmse double NOT NULL,
                UNIQUE (fireid, eps_alpha, eps_beta)
            )
        ''')
        conn.commit() # Commit the changes to the database

    except sqlite3.Error as e:
        print(f"Error connecting or creating table: {e}")
    finally:
        if conn:
            conn.close() # Close the database connection

def insert_data(db_name, fireid: int, eps_alpha, eps_beta, rmse):
    """
    Connects to an SQLite database and inserts a new record into the 'users' table.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Insert a new record into the 'users' table.
        cursor.execute("INSERT INTO users (fireid, eps_alpha, eps_beta, rmse) VALUES (?, ?, ?, ?)", (fireid, eps_alpha, eps_beta, rmse))
        conn.commit() # Commit the changes

    except sqlite3.Error as e:
        print(f"Error inserting data: {e}")
    finally:
        if conn:
            conn.close()

def fetch_data(db_name: str, fireid, alpha, beta):
    """
    Connects to an SQLite database and fetches all records from the 'users' table.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Correctly format the SQL query with placeholders and pass parameters
        cursor.execute(
            "SELECT count(*) FROM users WHERE fireid = ? AND eps_alpha = ? AND eps_beta = ?",
            (fireid, alpha, beta)
        )
        count = cursor.fetchone()[0] # Fetch the single count value
        
        if count == 1:
            return True
        else:
            return False

    except sqlite3.Error as e:
        raise ValueError(f"Error fetching data: {e}")
    finally:
        if conn:
            conn.close()

# create database to record rmse of ensf
# mainly for fine tune and avoid duplicated running the code
database_file = "ensf_final_rmse.db" # Define your database file name
create_database_and_table(database_file)

# alpha_t(0) = 1
# alpha_t(1) = esp_alpha \approx 0
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
def reverse_SDE(x0, n_dim, score_likelihood=None, time_steps=100,
                drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,  save_path=False):
    # x_T: sample from standard Gaussian
    # x_0: target distribution to sample from

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
    
# farsite system
fireid = 2286
SDE_sigma = 1

# filtering setup
dt = 12/24

####################################################################
# EnSF setup
obs_sigma = 0.5
# define the diffusion process
# eps_alpha_list = [0.96]#[0.75, 0.5, 0.25, 0.1]
# eps_beta_list = [0.03]#[0.8, 0.99, 0.95, 0.6]
eps_alpha, eps_beta = 0.96, 0.03

# ensemble size
ensemble_size = 30

# forward Euler step
euler_steps = 200

# damping function(tau(0) = 1;  tau(1) = 0;)
g_tau = lambda t: 1-t

landscape = rf"{fireid}/{fireid}.lcp"

# computation setting
torch.set_default_dtype(torch.float64) # half precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

workers = max(int(os.cpu_count() / 2 + 1), 20)

# for eps_alpha in eps_alpha_list:
#     for eps_beta in eps_beta_list:
for fireid in [1819, 2286, 3128, 3173]:
    indicator = fetch_data(database_file, fireid, eps_alpha, eps_beta)
    print(fireid, eps_alpha, eps_beta)
    # if indicator:
    #     continue

    torch.manual_seed(114514)
    filtering_steps = np.sort([int(s) for s in os.listdir(str(fireid)) if "." not in s])[:-1]
    ####################################################################
    ####################################################################
    # filtering initial ensemble
    x_state = gpd.read_file(rf"{fireid}/0/fire_boundary.shp", engine='pyogrio')
    x_state = torch.tensor(x_state.loc[0, 'geometry'].exterior.coords, device=device).flatten()
    n_dim = len(x_state)
    x_state = x_state.repeat(ensemble_size, 1) + 0.1 * torch.randn(ensemble_size, n_dim, device=device)

    torch.cuda.empty_cache()

    tme = 0

    # info containers
    rmse_all = [0]
    obs_save = []
    est_save = []

    # filtering cycles
    for i in filtering_steps:
        check_point = time()

        # get observation
        obs = gpd.read_file(rf"{fireid}/{i+1}/fire_boundary.shp", engine='pyogrio')
        obs = np.array(obs.loc[0, 'geometry'].exterior.coords)

        ens_loc = rf"{fireid}/{i}/ensemble"
        create_folder(ens_loc)
        input = rf"{fireid}/{i}/para.input"

        run_list = []
        for j in range(ensemble_size):
            each_loc = ens_loc + rf'/{j}'
            create_folder(each_loc)
            ens_tensor = x_state[j].reshape(-1,2).cpu().numpy()
            ens_poly = Polygon(ens_tensor)
            ens_gpd = gpd.GeoDataFrame(geometry=[ens_poly])
            gpd_loc = rf"{each_loc}/state.shp"
            ens_gpd.to_file(gpd_loc)

            output_loc = rf"{each_loc}/out"
            
            ins_file = farsite_ins(landscape, input, gpd_loc, output_loc)
            
            run_list.append(ins_file)
            
        # prediction step ############################################
        # state forward in time
        run_parallel(run_list, workers)
        
        ensem_list = [obs]
        size_list = [len(obs)]
        
        for j in range(ensemble_size):
            output_file_loc = ens_loc + rf'/{j}/out/output_Perimeters.shp'
            next_line = gpd.read_file(output_file_loc, engine='pyogrio')
            line = next_line.iloc[-1, -1]
            coords = np.array(line.coords)
            ensem_list.append(coords)
            size_list.append(len(ensem_list))
            
        max_dim = max(size_list)

        obs_and_state = []
        for j in range(ensemble_size + 1):
            obs_and_state.append(order_polygon_vertices_convex(resample_polygon(ensem_list[j], max_dim)).flatten())
        
        obs = torch.tensor(obs_and_state[0], device = device)

        x_state = torch.tensor(obs_and_state[1:], device = device)
        x_state += np.sqrt(dt)*SDE_sigma*torch.randn_like(x_state)

        # define likelihood score
        # obs: (d)
        # xt: (ensemble, d)
        score_likelihood = lambda xt, t: -(xt - obs) / obs_sigma**2 * g_tau(t)

        # generate posterior sample
        x_state = reverse_SDE(x0 = x_state, n_dim = int((max_dim + 1)* 2), score_likelihood=score_likelihood, time_steps=euler_steps)
        
        # get state estimates
        x_est = torch.mean(x_state,dim=0).reshape(-1, 2).cpu().numpy()
        obs = obs.reshape(-1, 2).cpu().numpy()

        x_est = order_polygon_vertices_convex(x_est)
        obs = order_polygon_vertices_convex(obs)
        
        # get rmse
        x_est = np.array(convert_WGS84(gpd.GeoDataFrame(geometry=[Polygon(x_est)]), fireid, i).loc[0, 'geometry'].exterior.coords)
        obs = np.array(convert_WGS84(gpd.GeoDataFrame(geometry=[Polygon(obs)]), fireid, i).loc[0, 'geometry'].exterior.coords)

        obs_save.append(obs)
        est_save.append(x_est)

        rmse_temp = compute_rmse(x_est, obs)

        tme += (time() - check_point)

        if x_state.device.type == 'cuda':
            torch.cuda.current_stream().synchronize() #Wait for all kernels in all streams on a CUDA device to complete.

        # if rmse_temp > 1000 or np.isnan(rmse_temp):
        #     print('diverge!')
        #     rmse_all = [0, 1000]
        #     break

        # save information
        rmse_all.append(rmse_temp)
        np.save(rf'{fireid}/ensf_{i}.npy', x_est)
    np.save(rf'{fireid}/ensf_rmse.npy', np.array(rmse_all))
    # insert_data(database_file, fireid, eps_alpha, eps_beta, np.mean(rmse_all[1:]))
    # save results
    # obs_save = np.vstack(obs_save)
    # est_save = np.vstack(est_save)
        
    # obs_save = torch.stack(obs_save, dim=0).cpu().numpy()
    # est_save = torch.stack(est_save, dim=0).cpu().numpy()

