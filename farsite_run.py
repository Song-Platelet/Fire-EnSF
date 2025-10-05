import os
import site
import requests
import warnings
import subprocess
from tqdm import tqdm
from time import time

import rasterio
import geopandas as gpd

from shapely.ops import unary_union, polygonize
from shapely.geometry import box, shape, Point, Polygon, MultiPoint, MultiPolygon, LineString, MultiLineString

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import *

if __name__ == "__main__":
    plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or 'Computer Modern Roman' for LaTeX feel
    plt.rcParams['font.size'] = 12  # Base font size
    plt.rcParams['axes.labelsize'] = 14 # Font size for x and y labels
    plt.rcParams['axes.titlesize'] = 16 # Font size for the title
    plt.rcParams['xtick.labelsize'] = 12 # Font size for x-axis tick labels
    plt.rcParams['ytick.labelsize'] = 12 # Font size for y-axis tick labels
    plt.rcParams['legend.fontsize'] = 12 # Font size for legend

    # Filter out only directories
    id_list = [1819, 2286, 3128, 3173]

    for fireid in id_list:
        period = np.sort([int(s) for s in os.listdir(str(fireid)) if "." not in s])
        landscape = rf"{fireid}/{fireid}.lcp"

        rmse_list = []
        for i in period[:-1]:
            output_loc = rf"{fireid}/{i}/out"
            input = rf"{fireid}/{i}/para.input"
            if i == 0:
                ig = rf"{fireid}/{i}/fire_boundary.shp"
            else:
                ig = rf"{fireid}/{i - 1}/out/output.shp"

            ins_file = farsite_ins(landscape, input, ig, output_loc)

            farsite(ins_file)
            next_line = gpd.read_file(rf"{fireid}/{i}/out/output_Perimeters.shp", engine='pyogrio')
        
            line = next_line.iloc[-1, -1]

            coords = list(line.coords)

            # Close the LineString if it's not already closed
            if not line.is_closed:
                coords.append(coords[0])  # Add the first point to the end

            # Create the Polygon
            final_polygon = Polygon(coords)

            est = gpd.GeoDataFrame(geometry=[final_polygon])
            est.to_file(rf"{fireid}/{i}/out/output.shp")

            obs = gpd.read_file(rf"{fireid}/{i+1}/fire_boundary.shp", engine='pyogrio')

            est = convert_WGS84(est, fireid, i)
            obs = convert_WGS84(obs, fireid, i)

            est_arr = np.array(est.loc[0, 'geometry'].exterior.coords)
            obs_arr = np.array(obs.loc[0, 'geometry'].exterior.coords)
                
            est_arr = order_polygon_vertices_convex(est_arr)
            obs_arr = order_polygon_vertices_convex(obs_arr)

            polygon_arrays = [est_arr, obs_arr]
            
            num_points = max(obs_arr.shape[0], est_arr.shape[0])

            resampled_polygons = []
            for arr in polygon_arrays:
                resampled = resample_polygon(arr, num_points)
                resampled_polygons.append(resampled)
            rmse_list.append(compute_rmse(order_polygon_vertices_convex(resampled_polygons[0]), order_polygon_vertices_convex(resampled_polygons[1])))

fig, ax = plt.subplots(figsize=(8, 5))
# print(len(rmse_list))
ax.plot(np.arange(1, len(rmse_list)+1),rmse_list, linewidth=1, color='dodgerblue', marker='s', markersize=6, markevery=2) # Added markers every 5 points
for i in range(0, len(rmse_list), 2):
# ax.text(x_coordinate, y_coordinate, text_label, ...)
# Adding a small offset to y for better visibility
    ax.text(i + 1, rmse_list[i] + 0.25, f'{rmse_list[i]:.2f}',
            ha='center', va='bottom', fontsize=12, color='dodgerblue')
ax.set_xlabel("Farsite Step", fontsize=plt.rcParams['axes.labelsize'])
ax.set_ylabel("RMSE \n Haversine Distance(km)", fontsize=plt.rcParams['axes.labelsize']) # Be more descriptive with y-label
ax.set_title("RMSE Over Farsite", fontsize=plt.rcParams['axes.titlesize'], fontweight='bold')

# Grid
ax.grid(True, linestyle='--', alpha=0.75, axis = 'y') # Customize grid style
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

# --- Show and Save the Plot ---
plt.savefig(rf'{fireid}/rmse.png')
np.save(rf'{fireid}/farsite_rmse.npy', np.array(rmse_list))
