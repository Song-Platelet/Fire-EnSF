from util import *

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString


plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or 'Computer Modern Roman' for LaTeX feel
plt.rcParams['font.size'] = 12  # Base font size
plt.rcParams['axes.labelsize'] = 14 # Font size for x and y labels
plt.rcParams['axes.titlesize'] = 16 # Font size for the title
plt.rcParams['xtick.labelsize'] = 12 # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Font size for y-axis tick labels
plt.rcParams['legend.fontsize'] = 12 # Font size for legend


def scd(plg1, plg2):
    plg1, plg2 = plg1.buffer(0), plg2.buffer(0)
    intersection_poly = plg1.intersection(plg2)
    dice_coefficient = (2 * intersection_poly.area) / (plg1.area + plg2.area)
    return dice_coefficient

def ensemble_to_plg(loc):
    arr = np.load(loc)
    arr_est = np.mean(arr, axis = 0)

    n_dim = int(arr.shape[1]/2)
    
    arr_est = arr_est.reshape(n_dim, 2)
    
    return Polygon(arr_est)

df_data = []
id_list = [1819, 2286]

x = np.arange(1, 13, 3)

for fireid in id_list:

    for i in x:
        obs_poly = gpd.read_file(rf"{fireid}/{i}/fire_boundary.shp", engine='pyogrio').loc[0, 'geometry']
        farsite_poly = gpd.read_file(rf"{fireid}/{i - 1}/out/output.shp", engine='pyogrio').loc[0, 'geometry']
        
        current_stat = [fireid, i]
        
        ensf_poly = ensemble_to_plg(f'resub/{fireid}/ensf_{i-1}.npy')
        enkf_poly = ensemble_to_plg(f'resub/{fireid}/enkf_{i-1}.npy')
        
        obs_list = [obs_poly] * 3
        compare_list = [ensf_poly, enkf_poly, farsite_poly]

        with ProcessPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(scd, obs_list, compare_list))

        current_stat += results
        df_data.append(current_stat)

df = pd.DataFrame(df_data, columns = ['fireid', 'period', 'Fire-EnSF', 'FARSITE-EnKF', 'FARSITE' ])

bar_width = 0.85

def lighten_color(hex_color, factor=0.5):
    """
    Lighten a hex color by mixing with white.
    factor = 0 -> original color, factor = 1 -> white.
    Typical factor 0.3–0.7 gives a pastel look.
    """
    rgb = mcolors.hex2color(hex_color)          # (R,G,B) in 0..1
    white = (1.0, 1.0, 1.0)
    lighter_rgb = [c + (white[i] - c) * factor for i, c in enumerate(rgb)]
    return mcolors.to_hex(lighter_rgb)

colors = ['#3850a1', '#e8252d', '#FF6BA7']
colors_light = [lighten_color(c, factor=0.6) for c in colors]

fig, axes = plt.subplots(nrows=2, ncols=1,
                         figsize=(8, 10),   # width, height per subplot
                         sharex=True,              # common x-axis (period)
                         tight_layout=True)        # avoid overlap

# -------------------------------
# 5. Plot each fire in its own row
# -------------------------------
for ax, fid in zip(axes, id_list):
    subset = df[df['fireid'] == fid]
    
    # Extract values for each variable, aligned with periods_unique
    var1_vals = subset['Fire-EnSF'].values
    var2_vals = subset['FARSITE-EnKF'].values
    var3_vals = subset['FARSITE'].values
    
    # Plot bars
    ax.bar(x - bar_width - 0.03, var1_vals, width=bar_width, label='Fire-EnSF', color=colors_light[0], edgecolor = colors[0], linewidth=1.3)
    ax.bar(x, var2_vals, width=bar_width, label='FARSITE-EnKF', color=colors_light[1], edgecolor = colors[1], linewidth=1.3)
    ax.bar(x + bar_width + 0.03, var3_vals, width=bar_width, label='FARSITE', color=colors_light[2], edgecolor = colors[2], linewidth=1.3)


    # Academic styling
    ax.set_xticks(x)                     # centre of each group
    ax.set_xticklabels([str(j) for j in x])
    ax.set_ylabel('Sorensen-Dice Coefficient', fontsize=11)
    ax.set_title(f'Fire ID: {fid}', fontsize=13, fontweight='bold')
    ax.grid(axis = 'y', linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=False)
    ax.tick_params(axis='both', labelsize=9)

# Bottom x‑axis label (shared)
axes[-1].set_xlabel('Filtering Step', fontsize=12)


# Save high‑resolution figure (academic)
plt.savefig(r'rev/scd.pdf', dpi=300, bbox_inches='tight')
plt.show()        

    
        
