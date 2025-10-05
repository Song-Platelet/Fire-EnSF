import os
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.ops import unary_union, polygonize
from shapely.geometry import shape, Point, Polygon, MultiPoint, MultiPolygon, LineString, MultiLineString

import torch
from util import *

# farsite system
fireid = 2286
SDE_sigma = 1

# filtering setup
dt = 12/24
filtering_steps = np.sort([int(s) for s in os.listdir(str(fireid)) if "." not in s])[:-1]

####################################################################
# EnSF setup
obs_sigma = 0.5

# ensemble size
ensemble_size = 30

landscape = rf"{fireid}/{fireid}.lcp"

# computation setting
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

workers = max(int(os.cpu_count() / 2 + 1), 20)

torch.manual_seed(114514)

x_state = gpd.read_file(rf"{fireid}/0/fire_boundary.shp", engine='pyogrio')
x_state = torch.tensor(x_state.loc[0, 'geometry'].exterior.coords, device=device).flatten()
n_dim = len(x_state)

torch.cuda.empty_cache()

x_state = x_state.repeat(ensemble_size, 1) + 0.1 * torch.randn(ensemble_size, n_dim, device=device)

rmse_kf = [0]

t = 0

for i in filtering_steps:
    check_point = time()

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
    n_dim = int((max_dim + 1)* 2)

    obs_and_state = []
    for j in range(ensemble_size + 1):
        obs_and_state.append(order_polygon_vertices_convex(resample_polygon(ensem_list[j], max_dim)).flatten())
    
    obs = torch.tensor(obs_and_state[0], device = device)

    x_state = torch.tensor(obs_and_state[1:], device = device)
    x_state += np.sqrt(dt)*SDE_sigma*torch.randn_like(x_state)
    
    x_mean = torch.mean(x_state, axis=0)
    e_x = x_state - x_mean

    y_perturbed = obs.unsqueeze(0).repeat(ensemble_size, 1)
    y_perturbed += obs_sigma * torch.randn(ensemble_size, n_dim, device=device)
    
    Hx = x_state  # 应用观测算子 H = 0.25*I
    Hx_mean = torch.mean(Hx, axis=0)
    e_Hx = Hx - Hx_mean
    
    P_yy = e_Hx.T @ e_Hx / (ensemble_size - 1) + torch.eye(n_dim, device=device) * (obs_sigma**2)
    P_xy = e_x.T @ e_Hx / (ensemble_size - 1)

    K = P_xy @ torch.linalg.inv(P_yy)

    x_state = x_state + (y_perturbed - Hx) @ K.T
    
    x_est = torch.mean(x_state, axis=0).reshape(-1, 2).cpu().numpy()
    t += time() - check_point
    obs = obs.reshape(-1, 2).cpu().numpy()

    x_est = order_polygon_vertices_convex(x_est)
    obs = order_polygon_vertices_convex(obs)

    x_est = np.array(convert_WGS84(gpd.GeoDataFrame(geometry=[Polygon(x_est)]), fireid, i).loc[0, 'geometry'].exterior.coords)
    obs = np.array(convert_WGS84(gpd.GeoDataFrame(geometry=[Polygon(obs)]), fireid, i).loc[0, 'geometry'].exterior.coords)

    rmse = compute_rmse(x_est, obs)
    rmse_kf.append(rmse.item())

    if x_state.device.type == 'cuda':
        torch.cuda.current_stream().synchronize()

    np.save(rf'{fireid}/enkf_{i}.npy', x_est)

print('done')
rmse_kf = np.array(rmse_kf)
np.save(rf'{fireid}/enkf_rmse.npy', rmse_kf)
# est_save = torch.stack(est_save, dim=0).cpu().numpy()