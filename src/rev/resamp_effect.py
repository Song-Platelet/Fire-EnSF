from util import *

from time import time
from pathlib import Path
import concurrent.futures
from functools import partial

import numpy as np
import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Ellipse

import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString

plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or 'Computer Modern Roman' for LaTeX feel
plt.rcParams['font.size'] = 12  # Base font size
plt.rcParams['axes.labelsize'] = 14 # Font size for x and y labels
plt.rcParams['axes.titlesize'] = 16 # Font size for the title
plt.rcParams['xtick.labelsize'] = 12 # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Font size for y-axis tick labels
plt.rcParams['legend.fontsize'] = 12 # Font size for legend

# Filter out only directories
fireid = 3173
i = 3

obs_loc = rf"{fireid}/{i}/fire_boundary.shp"
obs = gpd.read_file(obs_loc, engine='pyogrio')
obs = convert_WGS84(obs, fireid, i).loc[0, 'geometry']

def com_plot(x1, y1, x2, y2, file_loc):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax1, ax2 = axes

    # Plot the outlines and fill them with semi-transparent colors (alpha)
    ax1.plot(x1, y1, marker = 'o', color='blue')
    ax1.fill(x1, y1, alpha=0.3, color='blue', label='Original')
    ax1.set_title(f"Number of vertice: {int(len(x1))}")

    ax2.plot(x2, y2, marker = 'o', color='red')
    ax2.fill(x2, y2, alpha=0.3, color='red', label='After re-interpolated')
    ax2.set_title(f"Number of vertice: {int(len(x2))}")

    # ax1.set_aspect('equal') # Ensures squares look like squares
    ax1.grid(True, linestyle='--')
    ax1.legend()
    ax1.set_ylabel('Latitude')

    ax2.grid(True, linestyle='--')
    ax2.legend()

    # Sync the axis limits so both plots share the same scale
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    # --- Formatting: Apply directly to ax1 and ax2 ---
    for ax in [ax1, ax2]:
        ax.set_xlabel('Longitude')

        ax.grid(True, linestyle='--')
        ax.legend()
        
        # Configure locators and formatters on the specific axis object
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Display the plots
    plt.tight_layout()
    plt.savefig(file_loc)

x1, y1 = obs.exterior.xy
resamp = resample_polygon(np.array(obs.exterior.coords), 400)

# Extract x and y coordinates from the exterior of the polygons
x1, y1 = obs.exterior.xy
x2, y2 = resamp[:, 0], resamp[:, 1]

com_plot(x1, y1, x2, y2, r'rev/before_farsite.pdf')

resamp = Polygon(resamp)
intersection_poly = obs.intersection(resamp)

dice_coefficient = (2 * intersection_poly.area) / (obs.area + resamp.area)

with open(r'rev/SDC.txt', 'w') as f:
    f.write(f'Before FARSITE: {dice_coefficient:.4f}\n')


input = rf"{fireid}/{i}/para.input"

landscape = rf"{fireid}/{fireid}.lcp"

obs_loc = rf"{fireid}/{i}/fire_boundary.shp"
obs = gpd.read_file(obs_loc, engine='pyogrio')

resamp = resample_polygon(np.array(obs.loc[0, 'geometry'].exterior.coords), 400)
resamp = gpd.GeoDataFrame(geometry=[Polygon(resamp)])

obs.to_file(r'rev/obs.shp')
resamp.to_file(r'rev/resamp.shp')

def farsite_ins(landscape, input, ig, output_loc):
    create_folder(output_loc)
    output_loc = output_loc + '/output'
    template = rf"/home/hshi301/{landscape} /home/hshi301/{input} /home/hshi301/{ig} 0 /home/hshi301/{output_loc} 0"

    run_loc = 'rev/run.txt'

    with open(run_loc, 'w') as f:
        f.write(template)
    return rf"/home/hshi301/{run_loc}"

poly_list = []

# Current date and time
now = datetime.now()
print(now)


# 1. Define the worker function
def process_shapefile(ig, landscape, input_data, fireid, i):
    t1 = time()
    output_loc = rf"{ig[:-4]}/out"

    # Run your FARSITE simulation
    ins_file = farsite_ins(landscape, input_data, ig, output_loc)
    farsite(ins_file)

    # Read output and process geometry
    next_line = gpd.read_file(rf"{output_loc}/output_Perimeters.shp", engine='pyogrio')
    line = next_line.iloc[-1, -1]
    coords = list(line.coords)

    # Close the LineString if it's not already closed
    if not line.is_closed:
        coords.append(coords[0])  # Add the first point to the end

    # Create the Polygon and GeoDataFrame
    final_polygon = Polygon(coords)
    final_gdf = gpd.GeoDataFrame(geometry=[final_polygon])

    # Convert coordinates
    final_gdf = convert_WGS84(final_gdf, fireid, i)
    print(f'{ig}\nTime:   {time() - t1:.2f}')
    # Return the processed GeoDataFrame
    return final_gdf

ig_list = [r'rev/obs.shp', r'rev/resamp.shp']

# 2. Setup the parallel pool
with concurrent.futures.ProcessPoolExecutor() as executor:
    
    # 'partial' locks in the arguments that stay the same for every run
    worker = partial(
        process_shapefile, 
        landscape=landscape, 
        input_data=input, # passed your original 'input' variable here
        fireid=fireid, 
        i=i
    )

    results = executor.map(worker, ig_list)
            
    # Convert the results generator into your final list
    poly_list = list(results)

o1, r1 = poly_list[0].loc[0, 'geometry'].buffer(0), poly_list[1].loc[0, 'geometry'].buffer(0)

com_plot(x1, y1, x2, y2, r'rev/after_farsite.pdf')

intersection_poly = o1.intersection(r1)
dice_coefficient = (2 * intersection_poly.area) / (o1.area + r1.area)

with open(r'rev/SDC.txt', 'a') as f:
    f.write(f'After FARSITE: {dice_coefficient:.4f}')