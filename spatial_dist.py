import os
from util import *
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import geopandas as gpd
from shapely.geometry import box, shape, Point, Polygon

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter


plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or 'Computer Modern Roman' for LaTeX feel
plt.rcParams['font.size'] = 12  # Base font size
plt.rcParams['axes.labelsize'] = 18 # Font size for x and y labels
plt.rcParams['axes.titlesize'] = 20 # Font size for the title
plt.rcParams['xtick.labelsize'] = 12 # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Font size for y-axis tick labels
plt.rcParams['legend.fontsize'] = 12 # Font size for legend


id_list = [1819, 2286, 3128, 3173]
method_list = ['ensf', 'enkf']
method_dict = {'ensf':'EnSF', 'enkf':'EnKF'}
filter_steps = np.arange(1, 13, 3)

def spatial_plot(ax, fireid, method, i, final_ensemble, mean_kw=None, obs_kw=None):
    if mean_kw is None:
        mean_kw = {'color': 'darkgreen', 'linestyle': '--', 'linewidth': 1.25, 'label': f'Estimation'}
    if obs_kw is None:
        obs_kw = {'color': '#e8252d', 'linewidth': 1.25, 'label': 'Observation'}

    ensemble = final_ensemble[1:]
    
    for member in ensemble:
        member = member.reshape(-1, 2)
        ax.plot(member[:, 0], member[:, 1], 
                color='gray', alpha=0.2, linewidth=0.7)

    est = np.mean(ensemble, axis=0).reshape(-1, 2)
    ax.plot(est[:, 0], est[:, 1], **mean_kw)

    obs = final_ensemble[0].reshape(-1, 2)
    ax.plot(obs[:, 0], obs[:, 1], **obs_kw)

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.ticklabel_format(style='plain', useOffset=False, axis='both')
    ax.set_xlabel(f'Longtitude')
    if i == 1:
        ax.set_ylabel(f'{fireid} - {method_dict[method]}\nLatitude')

def process_member(member, fireid, i):
    """Process one ensemble member and return its coordinates as a numpy array."""
    member = member.reshape(-1, 2)
    member_polygon = Polygon(member)
    member_df = gpd.GeoDataFrame(geometry=[member_polygon])
    member_df = convert_WGS84(member_df, fireid, i)   # user-defined function
    coords = member_df.loc[0, 'geometry'].exterior.coords
    return np.array(coords)

for fireid in id_list:
    for m in method_list:
        if fireid > 3000 and m == 'enkf':
            break

        fig, axes = plt.subplots(nrows=1, ncols=len(filter_steps),
                                figsize=(20, 5),   # width, height per subplot
                                sharey=True,              # common x-axis (period)
                                tight_layout=True)        # avoid overlap

        for j, i in enumerate(filter_steps):
            
            file_loc = rf'resub/{fireid}/{m}_{i-1}.npy'
            file_loc = os.path.abspath(file_loc)
            ensemble = np.load(file_loc)
            

            obs = gpd.read_file(rf"{fireid}/{i}/fire_boundary.shp", engine='pyogrio')
            obs = np.array(obs.loc[0, 'geometry'].exterior.coords).flatten()

            final_ensemble = [process_member(obs, fireid, i)]

            # Process ensemble members in parallel
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_member, member, fireid, i) for member in ensemble]
                member_results = [f.result() for f in futures]

            final_ensemble += member_results
                
            spatial_plot(axes[j], fireid, m, i, final_ensemble)

            if m == 'ensf' and fireid == 1819:
                axes[j].set_title(f'Filtering Step {i}')
            
        plt.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
        fig.subplots_adjust(right=0.82)   
        plt.savefig(rf'resub/{fireid}_{m}.pdf', dpi=300, bbox_inches='tight')



