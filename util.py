import os
import glob
import site
import shutil
import zipfile
import warnings
warnings.filterwarnings('ignore')

import subprocess
from time import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors

import fiona
from pyproj import CRS
import rasterio
from rasterio.plot import show as raster_show
import geopandas as gpd
from shapely.ops import unary_union, polygonize
from shapely.geometry import box, shape, Point, Polygon, MultiPoint, MultiPolygon, LineString, MultiLineString

def farsite(instr_file, exe_path=r"./src/TestFARSITE"):
    """
    Run FARSITE with input files.
    
    Args:
        instr_file (str): Instruction of input for FARSITE to operate
        exe_path (str, optional): Path to FARSITE executable. Uses default if None.
    """
    start = time()
    result = subprocess.run(
        [exe_path, instr_file],
        capture_output=True,
        text=True
    )
    elapsed = time() - start
    return {
        'input_file': instr_file,
        'time': elapsed,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    }

def run_parallel(input_files:list, max_workers=None, exe_path=None):
    """
    Run FARSITE in parallel for multiple input files.
    
    Args:
        input_files (list): List of input file paths.
        max_workers (int, optional): Maximum number of parallel executions. Defaults to None.
        exe_path (str, optional): Path to FARSITE executable. Uses default if None.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare futures with or without custom exe_path
        futures = {}
        for file in input_files:
            if exe_path:
                future = executor.submit(farsite, file, exe_path)
            else:
                future = executor.submit(farsite, file)
            futures[future] = file

def create_folder(folder_path):
    """
    Creates a folder at the specified path.

    Args:
        folder_path (str): The path where the folder should be created.
                           This can be a relative or absolute path.
    """
    try:
        # Check if the folder already exists
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)
    except OSError as error:
        print(f"Error creating folder '{folder_path}': {error}")

def farsite_ins(landscape, input, ig, output_loc):
    """
    Creates a instruction file for FARSITE to run

    Args:
        landscape (str): landscape file location
        input (str): location of .input file
        ig (str): location of initial fire boundry file
        output_loc (str): the folder for FARSITE to save the output
    
    Returns:
        The location of instruction file
    """
    create_folder(output_loc)
    output_loc = output_loc + '/output'
    template = rf"{os.path.abspath(landscape)} {os.path.abspath(input)} {os.path.abspath(ig)} 0 {os.path.abspath(output_loc)} 0"

    with open(output_loc[:-10] + 'run.txt', 'w') as f:
        f.write(template)
    return os.path.abspath(rf"{output_loc[:-10]}run.txt")

def order_polygon_vertices_convex(points):
    """
    Orders a pre-sequenced list of 2D polygon points clockwise,
    starting from the point closest to the positive x-axis.

    This function assumes the input 'points' already defines the
    polygon's perimeter. It will check the winding order, reverse it
    if necessary, and then rotate the list to the correct starting point.

    Args:
        points: A list of tuples, where each tuple is an (x, y) coordinate
                representing the connected vertices of a polygon.

    Returns:
        A new list of points, ordered clockwise from the correct start.
    """
    points_arr = np.array(points)
    n = len(points_arr)
    if n < 3:
        return points_arr
    
    x = points_arr[:, 0]
    y = points_arr[:, 1]
    
    next_y = np.roll(y, -1)
    next_x = np.roll(x, -1)
    signed_area = 0.5 * np.sum(x * next_y - y * next_x)
    
    if signed_area > 0:
        corrected_points = points_arr[::-1]
    else:
        corrected_points = points_arr.copy()
    
    x_pts = corrected_points[:, 0]
    y_pts = corrected_points[:, 1]
    angles = np.abs(np.arctan2(y_pts, x_pts))
    start_index = np.argmin(angles)
    
    ordered_points = np.roll(corrected_points, -start_index, axis=0)
    ordered_points = np.vstack([ordered_points, ordered_points[0]])
    
    return ordered_points

def resample_polygon(polygon_coords, num_points):
    """
    Resamples a polygon to a specific number of points.

    Args:
        polygon_coords (np.array): Array of shape (M, 2) representing
                                   the polygon vertices [x, y].
        num_points (int): The desired number of points in the resampled polygon.

    Returns:
        np.array: Array of shape (num_points, 2) representing the
                  resampled polygon vertices.
    """
    # Ensure the polygon is closed for distance calculation by appending the first point
    closed_polygon_coords = np.vstack([polygon_coords, polygon_coords[0]])

    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(closed_polygon_coords, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0) # Add 0 for the start point
    total_perimeter = cumulative_distances[-1]

    if total_perimeter == 0: # Handle cases of single point or zero area polygons
        return np.tile(polygon_coords[0], (num_points, 1))

    # Generate num_points along the perimeter [0, total_perimeter]
    # The last point will coincide with the first if the polygon is closed.
    target_distances = np.linspace(0, total_perimeter, num_points)

    # Interpolate x and y coordinates
    # np.interp needs monotonically increasing 'xp' (cumulative_distances)
    interp_x = np.interp(target_distances, cumulative_distances, closed_polygon_coords[:, 0])
    interp_y = np.interp(target_distances, cumulative_distances, closed_polygon_coords[:, 1])

    # Stack them back into (N, 2) array
    resampled_polygon = np.vstack([interp_x, interp_y]).T

    # The linspace and interpolation on closed_polygon_coords should ensure
    # the first and last points of resampled_polygon are the same if num_points > 1.
    return resampled_polygon

def haversine_distance(array1, array2):
    """
    Compute the Haversine distance between corresponding points in two arrays.
    
    Parameters:
    array1 (np.ndarray): First array of shape (n, 2) [lon, lat] in degrees.
    array2 (np.ndarray): Second array of shape (n, 2) [lon, lat] in degrees.
    
    Returns:
    np.ndarray: Distances between corresponding points in meters.
    """
    # Convert degrees to radians
    lon1, lat1 = np.radians(array1[:, 0]), np.radians(array1[:, 1])
    lon2, lat2 = np.radians(array2[:, 0]), np.radians(array2[:, 1])
    
    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine formula components
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Earth radius in kilometers (mean radius = 6371 km)
    R = 6371
    return R * c

def compute_rmse(array1, array2):
    """
    Compute RMSE for geographic coordinates using Haversine distance.
    
    Parameters:
    array1 (np.ndarray): First array of shape (n, 2).
    array2 (np.ndarray): Second array of shape (n, 2).
    
    Returns:
    float: RMSE in meters.
    """
    distances = haversine_distance(array1, array2)
    return np.sqrt(np.mean(distances**2))

def convert_WGS84(gdf, fireid, period):
    """
    Convert the project of shape to Geographic Coordinate System (longtitude and latitude).
    
    Parameters:
    gdf (geopandas.GeoDataFrame): First array of shape (n, 2).
    fireid (int): The id of fire.
    period (int): Current period.
    
    Returns:
    geopandas.GeoDataFrame: the polygon in GCS projection.
    """
    with open(rf'{fireid}/{period}/fire_boundary.prj', 'r') as f:
        prj_wkt = f.readline()
    gdf = gdf.set_crs(prj_wkt, allow_override=True)
    gdf = gdf.to_crs(epsg=4326)
    return gdf