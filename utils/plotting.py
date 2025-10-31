import os
import glob
import rioxarray
import shutil
import rasterio

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from matplotlib import cm
from matplotlib.colors import Normalize 
from matplotlib.animation import FuncAnimation
from IPython.display import Image, display
from datetime import datetime
from tqdm import tqdm

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
from collections import OrderedDict
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm
from kgcpy import *
from geopy.geocoders import Nominatim
from collections import defaultdict
from kgcpy import *
from geopy.geocoders import Nominatim
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

matplotlib.rcParams['agg.path.chunksize'] = 10000


def mae(pred, truth):

    """
    Function for calculating the mean absolute error (MAE) between the prediction and target.
    Args:
        pred: xarray.Dataset
            An xarray dataset of the model prediction.
        truth: xarray.Dataset
            An xarray dataset of the ground truth.
    """

    abs_diff = abs(pred-truth)
    mae = abs_diff.mean(dim=['x','y'], skipna=True)['band_data'].values[0]
    return mae

def norm(arr1):

    """
    Function for normalizing an array using histogram normaliztion.
    Args:
        arr1: numpy.ndarray()
        An array that contains un-normalized data.

    Returns:
        The normalized array.
    
    If np.nanmax(arr1) - np.nanmin(arr1) is 0:
        Returns an array of zeros
    """
    if np.nanmax(arr1) - np.nanmin(arr1) == 0: 
      return np.zeros_like(arr1)  # Avoid division by zero
    return (arr1 - np.nanmedian(arr1)) / np.nanstd(arr1)

def stack_rgb(stacked_inputs_path):
    """
    Extract, normalize and stack HLS bands corresponding to RGB image from stacked input file 
    Args:
        stacked_inputs_path (str): Path to stacked input file        
    """
    with rasterio.open(stacked_inputs_path) as src:
        blue_band = src.read(1)
        green_band = src.read(2)
        red_band = src.read(3)
        # normalize these bands
        red_normalized = norm(red_band) 
        green_normalized = norm(green_band) 
        blue_normalized = norm(blue_band) 
        rgb_normalized = np.dstack(((red_normalized + 0.3), (green_normalized +0.3), (blue_normalized +0.3)))
        rgb_normalized = xr.DataArray(rgb_normalized)
    
    return rgb_normalized


def calculate_metrics(target_fps, predict_fps):

    """
    Function for calculating some metrics given targets and predictions
    Args:
        target_fps: List of paths to LST ground-truth files.
        predict_fps: List of paths to predicted LST prediction.
    """

    metrics = []
    for tfp, pfp in tqdm(zip(target_fps, predict_fps), total=len(target_fps)):

        if not (os.path.isfile(tfp) and os.path.isfile(pfp)):
            print(f"Skipping:  Either one of these files missing: \n{tfp}\n{pfp}")
            continue

        # read in target data
        ds_target = xr.open_dataset(tfp)
        ds_target = ds_target.where(ds_target['band_data'] != -9999., np.nan)
        ds_pred = xr.open_dataset(pfp)
        ds_pred = ds_pred.where(~np.isnan(ds_target['band_data']), np.nan)

        # calculate MAE
        mae_val = mae(ds_pred, ds_target)

        if "index" in os.path.basename(tfp):
            city_country, _, _, datestamp, timestamp = os.path.basename(tfp).split(".")[0:5]
        else:
            city_country, _, datestamp, timestamp = os.path.basename(tfp).split(".")[0:4]

        timestamp = datetime.strptime(datestamp+timestamp, "%Y%m%dT%H%M%S")
        metrics.append((tfp, pfp, city_country, timestamp, mae_val))

    df = pd.DataFrame(metrics, columns=["target_fp", "prediction_fp", "city_country", "timestamp", "mae"])

    return df

def plot_box_plot(targets_path_patch, predictions_path_patch, results_path, save_plot):

    """Plotting routine
    Args:
        targets_path_patch (str): Path to directory with geotiff target patches.
        predictions_path_patch (str): Path to directory where the geotiff predictions are saved.
        result_path (str): Path to directory where the box plot must be saved.
        save_plot (bool): "True" will display and save the generated plot. "False" will only display the plot.
    """

    assert targets_path_patch.exists(), f"Folder not found: {str(targets_path_patch)}" 
    assert predictions_path_patch.exists(), f"Folder not found: {str(predictions_path_patch)}" 
    assert results_path.exists(), f"Folder not found: {str(results_path)}" 

    # gather target file paths and corresponding predictions file paths
    target_fps = glob.glob(os.path.join(targets_path_patch, "*.tif"))
    target_fps_names = [".".join(os.path.basename(x).split(".")[0:5]) for x in target_fps]
    predict_fps = [os.path.join(predictions_path_patch, f"{x}.inputs_pred.tif") for x in target_fps_names]

    # generate metrics df
    metrics_df = calculate_metrics(target_fps, predict_fps)

    means = np.squeeze(metrics_df.groupby('city_country').mean(numeric_only=True).values)
    stds = np.squeeze(metrics_df.groupby('city_country').std(numeric_only=True).values)
    names = list(np.squeeze(metrics_df.groupby('city_country').mean(numeric_only=True).index.values))
    xs = np.arange(len(means))

    plt.figure(figsize=(8, 6))
    plt.bar(xs, means, color='lightblue')
    plt.grid(True, alpha=0.5, zorder=0)
    plt.title('Box plots for unseen timestamps and cities')
    plt.xlabel('City')
    plt.ylabel('Mean absolute error ($^\circ$C)')
    plt.xticks(rotation=45)
    plt.errorbar(xs, means, stds, fmt='.', color='0.5', elinewidth=1.5, capthick=1.5, errorevery=1, ms=5, capsize=4)
    #plt.gca().set_xticklabels(names)
    plt.gca().set_xticks(np.arange(len(names)), labels=names)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.show()

    return metrics_df


def plot_results(tile_name, inputs_path, targets_path, predictions_path):
    """Plotting routine
        Args:
            tile_name:  Name of the tile (excluding suffix such as ".stacked_inputs.tif", ".lst.tif", and ".nputs_pred.tif")
            inputs_path: Path to the folder containing input images (suffix: ".inputs.tif")
            targets_path: Path to the folder containing ground-truth LST observations (suffix: ".lst.tif")
            predictions_path: Path to the folder containing the LST predictions (suffix: ".inputs_pred.tif")
    """
    
    input_path = inputs_path.joinpath(f"{tile_name}.inputs.tif")
    target_path = targets_path.joinpath(f"{tile_name}.lst.tif")
    pred_path = predictions_path.joinpath(f"{tile_name}.inputs_pred.tif")

    assert input_path.exists(), f"File not found: {str(input_path)}"
    assert target_path.exists(), f"File not found: {str(target_path)}"
    assert pred_path.exists(), f"File not found: {str(pred_path)}"

    # Read inputs
    input_bands = []
    with rasterio.open(input_path) as src:
        input_bands.append(norm(src.read(3))+0.3) #red
        input_bands.append(norm(src.read(2))+0.3) #green
        input_bands.append(norm(src.read(1))+0.3) #blue
    rgb_normalized = xr.DataArray(np.dstack((input_bands[0], input_bands[1], input_bands[2])))

    # Read target and prediction files
    ds_lst_target = xr.open_dataset(target_path)
    ds_lst_target = ds_lst_target.where(ds_lst_target['band_data'] != -9999., np.nan)
    ds_lst_pred = xr.open_dataset(pred_path)
    ds_lst_pred = ds_lst_pred.where(~np.isnan(ds_lst_target['band_data']), np.nan)
    ds_error =  ds_lst_pred - ds_lst_target

    min_target = ds_lst_target.quantile(0.05)['band_data'].values
    max_target = ds_lst_pred.quantile(0.95)['band_data'].values
    min_error = ds_error.quantile(0.005)['band_data'].values
    max_error = ds_error.quantile(0.995)['band_data'].values
    error_max_size = max(abs(min_error), max_error)
    min_error, max_error = -abs(error_max_size), error_max_size
    data_normalizer = Normalize(min_target, max_target)
    error_normalizer = Normalize(min_error, max_error)

    # calculate error metrics
    mae_val = round(mae(ds_lst_pred, ds_lst_target), 3)

    # Plotting
    nrows, ncols = 1, 5
    axw, axh = 4, 4
    wsp, hsp = 0.03, 0.03

    fig = plt.figure(figsize=(axw*ncols, axh*nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=wsp, hspace=hsp, width_ratios=[1,1,1,1,0.75])

    #Plot the RGB image
    ax_rgb = plt.subplot(gs[0, 0])
    plt.imshow(rgb_normalized, aspect="auto")
    ax_rgb.set_title("RGB Color Composite", fontsize=10)
    ax_rgb.get_xaxis().set_ticklabels([])
    ax_rgb.get_yaxis().set_ticklabels([])
    
    #Plot the target/ground truth
    ax_gt = plt.subplot(gs[0, 1])
    target_handle = ds_lst_target['band_data'].plot(ax=ax_gt, cmap='jet', norm=data_normalizer, vmin=min_target, vmax=max_target, add_colorbar=False)
    ax_gt.set_title("LST Observation", fontsize=10)
    ax_gt.set(ylabel=None, xlabel=None)
    ax_gt.get_xaxis().set_ticklabels([])
    ax_gt.get_yaxis().set_ticklabels([])

    #Plot the prediction
    ax_pred = plt.subplot(gs[0, 2])
    ds_lst_pred['band_data'].plot(ax=ax_pred, cmap='jet', norm=data_normalizer, vmin=min_target, vmax=max_target, extend='both', add_colorbar=False)
    ax_pred.set_title(f"LST Prediction", fontsize=10)
    ax_pred.set(ylabel=None, xlabel=None)
    ax_pred.get_xaxis().set_ticklabels([])
    ax_pred.get_yaxis().set_ticklabels([])

    #Plot the error (with the MAE in the the Title) 
    ax_err = plt.subplot(gs[0, 3])
    error_handle = ds_error['band_data'].plot(ax=ax_err, cmap='bwr', norm=error_normalizer, vmin=min_error, vmax=max_error, add_colorbar=False)
    ax_err.set_title(f"Error (MAE: {mae_val:.3f}$^\circ$C)", fontsize=10)
    ax_err.set(ylabel=None, xlabel=None)
    ax_err.get_xaxis().set_ticklabels([])
    ax_err.get_yaxis().set_ticklabels([])

    # colorbars
    ax_cbar = plt.subplot(gs[0, 4])
    cbar_lst = plt.colorbar(target_handle, ax=ax_cbar, orientation='vertical', location='left', extend='both', pad=0.25, label="Temperature ($^\circ$C)")
    cbar_err = plt.colorbar(error_handle, ax=ax_cbar, orientation='vertical', location='left', extend='both', pad=0.25, label="Error ($^\circ$C)")
    cbar_lst.ax.yaxis.set_ticks_position('right')
    cbar_err.ax.yaxis.set_ticks_position('right')
    ax_cbar.axis('off')

    # title
    city_name = input_path.stem.split(".")[0]
    acq_date = input_path.stem.split(".")[2]
    acq_time = input_path.stem.split(".")[3]
    timestamp = datetime.strptime(acq_date+acq_time, "%Y%m%dT%H%M%S")
    city_name = (city_name.split("_")[0]).upper() + f" ({(city_name.split('_')[1]).upper()})"
    fig.suptitle(f"City: {city_name},  Date & Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n", y=1.05, fontsize=12)

    plt.show()

def plot_rgb_lst_distribution_scatter(patches_tiles, target_patches_path, inference_path, result_path, input_path, patch, save_plot):
    """Plotting routine
        Args:
            patches_tiles (list): List of input patches or tiles to generate comparison plots for.
            target_patches_path (str): Path to directory with geotiff target patches.
            inference_path (str): Path to directory where the geotiff predictions are saved.
            result_path (str): Path to directory where the result/comparison plots must be saved.
            input_path (str): Path to directory with geotiff input patches.
            patch (bool): For patches set to True. For tiles set to False.
            save_plot (bool): "True" will display and save the generated plots. "False" will only display the plots.s
    """
    
    # inputs to parse to the main method
    input_patches_paths = []

    for pt in patches_tiles:
        patches_ = os.path.join(input_path, pt)
        input_patches_paths.append(patches_)

    # predictions to parse to the main method
    pred_patches_paths = []

    for inp in patches_tiles:
        inp_ = os.path.basename(inp)
        inp_ = inp_.replace("inputs", "inputs_pred")
        inp_ = os.path.join(inference_path, inp_)
        pred_patches_paths.append(inp_)

    # targets to parse to the main method
    target_patches_paths = []

    for tar in patches_tiles:
        patches_ = os.path.join(target_patches_path, tar)
        patches_ = patches_.replace("inputs", "lst")
        target_patches_paths.append(patches_)
    target_patches_paths

    #Create Plots Save Directory
    if os.path.exists(result_path):
        print("Results directory exits!")
    else: 
        print("Creating Results directory...")
        os.makedirs(result_path) 
    #Subdirectory to store comparison plots
    comp_plots_path = os.path.join(result_path, 'comparison_plots')
    if os.path.exists(comp_plots_path):
        print("Comparison plots directory exits!")
    else: 
        print("Creating comparison plots directory...")
        os.makedirs(comp_plots_path)

    ncols = 6
    nexp = 1
    axw, axh = 4, 4
    wsp, hsp = 0.03, 0.03
    title_x_pos = 0.5
    title_y_pos = 1.0
    
    for inp, tar, pred in zip(input_patches_paths, target_patches_paths, pred_patches_paths):

        fig = plt.figure(figsize=(axw*ncols, axh*nexp))
        gs = gridspec.GridSpec(nexp+1, ncols, height_ratios=[2,0.5], width_ratios=[1,1,1,1,1,1], wspace=wsp, hspace=hsp)

        #Open target and prediction as datasets
        ds_target = xr.open_dataset(tar)
        ds_target = ds_target.where(ds_target['band_data'] != -9999., np.nan)
        ds_pred = xr.open_dataset(pred)
        ds_pred = ds_pred.where(~np.isnan(ds_target['band_data']), np.nan)
        #Compute the error between the prediction and target
        error =  ds_pred - ds_target

        #Create the RGB object
        hls_bands = []
        with rasterio.open(inp) as src:
            red_band = src.read(3)
            hls_bands.append(red_band)
            green_band = src.read(2)
            hls_bands.append(green_band)
            blue_band = src.read(1)
            hls_bands.append(blue_band)
        red_normalized = norm(red_band)
        green_normalized = norm(green_band)
        blue_normalized = norm(blue_band)
        rgb_normalized = np.dstack((red_normalized+0.3, green_normalized+0.3, blue_normalized+0.3))
        rgb_normalized = xr.DataArray(rgb_normalized)

        mae_val = mae(ds_pred, ds_target)
        mae_val = round(mae_val, 3)

        #Min and max for predictions and errors
        min_p = np.nanmin(np.squeeze(ds_pred['band_data'].values))
        max_p = np.nanmax(np.squeeze(ds_pred['band_data'].values))
        min_e = np.nanmin(np.squeeze(error['band_data'].values))
        max_e = np.nanmax(np.squeeze(error['band_data'].values))
        error_abs = max(abs(min_e), max_e)
        error_min, error_max  = -abs(error_abs), error_abs

        #min and max for ground truth 
        gt_min = np.nanmin(np.squeeze(ds_target['band_data'].values))
        gt_max = np.nanmax(np.squeeze(ds_target['band_data'].values))
        
        #min and max across predictions and ground truth
        vmin, vmax = np.minimum(gt_min,min_p), np.maximum(gt_max, max_p)
        
        expi = 0 
        
        #Plot the RGB image
        ax_rgb = plt.subplot(gs[expi, 0])
        plt.imshow(rgb_normalized)
        ax_rgb.set_title("RGB", x=title_x_pos, y=title_y_pos)
        ax_rgb.get_xaxis().set_ticklabels([])
        ax_rgb.get_yaxis().set_ticklabels([])

        #Plot the target/ground truth
        ax_gt = plt.subplot(gs[expi, 1])
        target_handle = ds_target['band_data'].plot(ax=ax_gt, cmap='jet', vmin=vmin, vmax=vmax, add_colorbar=False)
        ax_gt.set_title("Ground Truth", x=title_x_pos, y=title_y_pos)
        ax_gt.set(ylabel=None)
        ax_gt.set(xlabel=None)
        ax_gt.get_xaxis().set_ticklabels([])
        ax_gt.get_yaxis().set_ticklabels([])

        #Plot the prediction
        ax_pred = plt.subplot(gs[expi, 2])
        ds_pred['band_data'].plot(ax=ax_pred, cmap='jet', vmin=vmin, vmax=vmax, add_colorbar=False)
        ax_pred.set_title(f"Prediction", x=title_x_pos, y=title_y_pos)
        ax_pred.set(ylabel=None)
        ax_pred.set(xlabel=None)
        ax_pred .get_xaxis().set_ticklabels([])
        ax_pred .get_yaxis().set_ticklabels([])

        #Plot the error (with the MAE in the the Title) 
        ax_err = plt.subplot(gs[expi, 3])
        error_handle = error['band_data'].plot(ax=ax_err, cmap='bwr', vmin=error_min, vmax=error_max, add_colorbar=False)
        ax_err.set_title(f"Error (MAE: {mae_val})", x=title_x_pos, y=title_y_pos)
        ax_err.set(ylabel=None)
        ax_err.set(xlabel=None)
        ax_err.get_xaxis().set_ticklabels([])
        ax_err.get_yaxis().set_ticklabels([])

        #Plot the prediction and target histograms
        ax_hist = plt.subplot(gs[expi, 4])
        ds_target['band_data'].plot.hist(bins=30, range=[vmin, vmax], label='Ground Truth')
        ds_pred['band_data'].plot.hist(bins=30, range=[vmin, vmax], alpha=0.5, label='Prediction')
        ax_hist.set_title(f"Histogram", x=title_x_pos, y=title_y_pos)
        ax_hist.legend(prop={'size': 10})
        ax_hist.set(xlabel=r'LST [$\degree$ C]')
        #ax_hist.get_xaxis().set_ticklabels([])
        ax_hist.get_yaxis().set_ticklabels([])

        #Flatten the target and prediction dtatsets into arrays
        target_line = np.squeeze(ds_target['band_data'].values.reshape(-1))
        pred_line = np.squeeze(ds_pred['band_data'].values.reshape(-1))

        #Plot the scatter density plot
        ax_sd = plt.subplot(gs[expi, 5])
        ax_sd.scatter(target_line, pred_line, s=5, alpha=0.5)
        ax_sd.plot([vmin, vmax], [vmin, vmax], 'r-')
        ax_sd.set_xlim(vmin, vmax)
        ax_sd.set_ylim(vmin, vmax)
        ax_sd.yaxis.tick_right()
        ax_sd.yaxis.set_label_position("right")
        ax_sd.set_title(f"Scatter Plot", x=title_x_pos, y=title_y_pos)
        ax_sd.set(xlabel=r'LST Obs [$\degree$ C]')
        ax_sd.set(ylabel=r'LST Pred [$\degree$ C]')

        # title
        city_name = inp.split("/")[-1].split(".")[0]
        if patch == False:
            acq_date = inp.split("/")[-1].split(".")[2]
            acq_time = inp.split("/")[-1].split(".")[3]
        acq_date = inp.split("/")[-1].split(".")[3]
        acq_time = inp.split("/")[-1].split(".")[4]
        timestamp = datetime.strptime(acq_date+acq_time, "%Y%m%dT%H%M%S")
        city_name = (city_name.split("_")[0]).upper() + f" ({(city_name.split('_')[1]).upper()})"
        fig.suptitle(f"City: {city_name},  Date & Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n", y=1.05, fontsize=12)
        # add colorbar for taget and prediction
        ax_cbar = plt.subplot(gs[expi+1, 1:3])
        #gs = gridspec.GridSpec(nexp+1, ncols, height_ratios=[1,0.5], width_ratios=[1,1,1,1,1])
        fig.colorbar(target_handle, ax=ax_cbar, use_gridspec=False, fraction=0.5, shrink=0.5, orientation='horizontal')
        ax_cbar.axis('off')

        #add colorbar for error map
        ax_cbar_err = plt.subplot(gs[expi+1, 3])
        #gs = gridspec.GridSpec(nexp+1, ncols, height_ratios=[1,0.5], width_ratios=[1,1,1,1,1])
        fig.colorbar(error_handle, ax=ax_cbar_err, use_gridspec=False, fraction=0.5, orientation='horizontal')
        ax_cbar_err.axis('off')

        #Save the image
        if save_plot == True:
            fname = os.path.join(comp_plots_path, (os.path.splitext(os.path.basename(inp))[0:5][0]) + '_comp_plot_enhance.png')
            print(f"Saving plot ..... {fname}")
            fig.savefig(fname, dpi=400, bbox_inches='tight')
        else:
            continue

    #Display the image
    fig.show()


def load_lst(pred_file):
    """Loads lst prediction files and returns the prediction as a Dataset
        Args:
            pred_file (str): Full path to the geotiff of the lst prediction 
    """
    try:
        with rasterio.open(pred_file) as src:
            lst_data = src.read(1)
            if lst_data is None or np.all(np.isnan(lst_data)):
                print(f"No valid data in {pred_file}")
                return np.zeros((src.height, src.width))  # Return a blank frame
            return lst_data
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return None

def plot_rgb_and_lst(input_file, pred_files, minmax, output_file):
    """Plotting routine
        Args:
            input_file (list): List of input files (geotiffs) to generate animation.
            pred_files (list): List of prediction files (geotiffs) to generate animation.
            minmax (float):  Calculate the temporal min and max of the predictions using the analyze_rasters method
            output_file (str): Animation output filename. Must use the .gif extension. 
    """
    # Create a GridSpec for the layout
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # Three columns: RGB, LST, Colorbar

    # Plot RGB tile
    ax_rgb = fig.add_subplot(gs[0, 0])
    if input_file:
        rgb = stack_rgb(input_file)
        ax_rgb.imshow(rgb, aspect='equal', interpolation='none')
        ax_rgb.set_title("RGB Tile", fontdict={'color': 'black'})
        ax_rgb.set_ylabel(f"RGB plot: {os.path.basename(input_file)}", fontsize=12)
        ax_rgb.axis('off')
    else:
        print("No input files found.")

    # Animation of predictions
    ax_lst = fig.add_subplot(gs[0, 1])
    ax_lst.set_title("LST Animation", fontdict={'color': 'black'})

    if pred_files:
        initial_lst = load_lst(pred_files[0])
        img = ax_lst.imshow(initial_lst, norm=Normalize(minmax.global_min, minmax.global_max), cmap='jet', alpha=1)
        ax_lst.axis('off')

        # Create colorbar in its own subplot
        ax_cbar = fig.add_subplot(gs[0, 2])
        cbar = plt.colorbar(img, cax=ax_cbar)  # Assign colorbar to the specified axis
        cbar.set_label('LST ($^{0}C$)', rotation=0, labelpad=20)

        # Animation of the predicted LST
        def update(frame):
            lst = load_lst(pred_files[frame])  # Load LST
            if lst is not None:
                img.set_array(lst)
                ax_lst.set_title(f"LST {os.path.basename(pred_files[frame]).split('.')[2]}.{os.path.basename(pred_files[frame]).split('.')[3]}")
            return img,

        ani = FuncAnimation(fig, update, frames=len(pred_files), interval=500, blit=False, repeat=True)

        # Save the animation
        ani.save(output_file, writer='pillow', fps=1)
        print(f"Animation saved as {output_file}")
    else:
        print("No prediction files found.")

    plt.tight_layout() 

class RasterStats:
    """Class to hold global minimum and maximum raster statistics."""
    def __init__(self, global_min, global_max):
        self.global_min = global_min
        self.global_max = global_max

def get_rasters(pred_files):
    """Get raster data from a list of predicted files.
        Args:
            pred_files (list): List of prediction files (geotiffs) to generate animation.
    """
    band_data = [] 
    for gtfp in pred_files:
        try:
            with rasterio.open(gtfp) as src:
                for band_num in range(1, src.count + 1):
                    band_data.append(src.read(band_num))
        except Exception as e:
            print(f"Error reading {gtfp}: {e}")
    return np.array(band_data)

def calc_stats(band_data_all):
    """Calculate the global minimum and maximum based on 5th and 95th percentiles.
    Args:
        band_data_all: numpy.ndarray()
            An array that contains the data from all the prediction rastersk

    """
    if len(band_data_all.shape) == 3:
        band_data_all = np.expand_dims(band_data_all, axis=0)

    all_data_combined = band_data_all.reshape(-1)

    global_min = np.nanpercentile(all_data_combined, 5)
    global_max = np.nanpercentile(all_data_combined, 95)

    return global_min, global_max

def analyze_rasters(inference_path):
    """function to analyze rasters in the given inference path.
    Args:
        inference_path (str): Path to directory where the geotiff predictions are saved.
    """
    pred_files = glob.glob(os.path.join(inference_path, "*_pred.tif"))
    pred_files = sorted(pred_files)  # Sort by date

    if pred_files:
        raster_data = get_rasters(pred_files)
        print("Raster data shape:", raster_data.shape)

        global_min, global_max = calc_stats(raster_data)
        print("5th Percentile (Global Minimum Value):", global_min)
        print("95th Percentile (Global Maximum Value):", global_max)
        
        return RasterStats(global_min, global_max)
    else:
        print("No predicted files found.")
        return None
 

def plot_preprocessed_images(input_file, target_file):
    """Plotting RGB image from stacked inputs generated
    Args:
        input_file (str): Path to stacked input file 
        target_file (str): Path to processed target file
    """
# Plot RGB from stacked inputs

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    
    rgb = stack_rgb(input_file)
    ax_rgb = plt.subplot(1, 2, 1)
    plt.imshow(rgb, alpha=1, aspect='auto', interpolation='none')
    ax_rgb.set_title("RGB tile", x=0.5, y=1.0, fontdict={'color': 'black'})
    ax_rgb.set_ylabel(f"RGB plot: {input_file.split('/')[-1].split('.')[0] +'.'+ input_file.split('/')[-1].split('.')[2] +'.'+ input_file.split('/')[-1].split('.')[3]}")
    ax_rgb.get_xaxis().set_ticklabels([])
    ax_rgb.get_yaxis().set_ticklabels([]);
    
    # Plot lst from processed targets
    lst_band = xr.open_dataset(target_file)
    lst_band = lst_band.rio.reproject("EPSG:4326")
    lst_band['LST (Degrees Celsius)'] = lst_band['band_data']
    lst_band = lst_band.drop_vars(['band_data'])
    
    ax_lst = plt.subplot(1, 2, 2)
    lst_band['LST (Degrees Celsius)'].plot(ax=ax_lst, cmap='jet', add_colorbar=True, robust=True)
    ax_lst.set_title("LST tile", x=0.5, y=1.0, fontdict={'color': 'black'});
    fig.tight_layout()
    fig.show()


class advanced_plots:
    def __init__(self, input_file, target_file, predict_file):
        self.input_file = input_file
        self.target_file = target_file
        self.predict_file = predict_file

    def unique_cities_list(self):
        cities = []
        for i in range(len(os.listdir(self.target_file))):
            entry=(os.listdir(self.target_file)[i])
            name =entry.split('.')[0]
            cities.append(name)
        self.unique_cities =list(OrderedDict.fromkeys(cities))

    def monthly_sort(self):
        self.cities_data={}
        monthly_data = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        for i in tqdm(range(len(self.unique_cities))):
            city_names = []
            jan_tar = []
            feb_tar = []
            mar_tar = []
            apr_tar = []
            may_tar = []
            jun_tar = []
            jul_tar = []
            aug_tar = []
            sep_tar = []
            oct_tar = []
            nov_tar = []
            dec_tar = [] 
            self.months_tar = []
            for j in range(len(os.listdir(self.target_file))):
                entry=(os.listdir(self.target_file)[j])
                name =entry.split('.')[0]
                timeStamp = entry.split('.')[3]
                year = timeStamp[0:4]
                month = timeStamp[4:6]
                day = timeStamp[6:8]

                if name == self.unique_cities[i]:
                    if month == "01":
                        jan_tar.append(entry)
                    elif month == "02":
                        feb_tar.append(entry)
                    elif month == "03":
                        mar_tar.append(entry)
                    elif month == "04":
                        apr_tar.append(entry)
                    elif month == "05":
                        may_tar.append(entry)
                    elif month == "06":
                        jun_tar.append(entry)
                    elif month == "07":
                        jul_tar.append(entry)
                    elif month == "08":
                        aug_tar.append(entry)
                    elif month == "09":
                        sep_tar.append(entry)
                    elif month == "10":
                        oct_tar.append(entry)
                    elif month == "11":
                        nov_tar.append(entry)
                    else:
                        dec_tar.append(entry)

            self.months_tar.append(jan_tar)
            self.months_tar.append(feb_tar)
            self.months_tar.append(mar_tar)
            self.months_tar.append(apr_tar)
            self.months_tar.append(may_tar)
            self.months_tar.append(jun_tar)
            self.months_tar.append(jul_tar)
            self.months_tar.append(aug_tar)
            self.months_tar.append(sep_tar)
            self.months_tar.append(oct_tar)
            self.months_tar.append(nov_tar)
            self.months_tar.append(dec_tar)
                    
            self.cities_data.update({self.unique_cities[i]:self.months_tar}) 
 

    def mse(self, pred, truth):
        mse = (pred-truth)**2
        mse = mse.mean(dim=['x','y'], skipna=True)['band_data'].values[0]
        return mse
    
    def rolling_window_calculation(self):
        self.cities_monthly_mae={}
        self.cities_monthly_mse={}
        self.cities_monthly_rmse={}
        self.cities_rolling_avg_mae={}
        self.cities_rolling_avg_mse={}
        self.cities_rolling_avg_rmse={}
        self.scatter_data_targ = {}
        self.scatter_data_pred = {}
        self.t2m_data = {}
        for i in range(len(self.unique_cities)):
            store_mae = []
            store_mse = []
            store_rmse = []
            monthly_mae = []
            monthly_mse = []
            monthly_rmse = []
            rolling_avg_mae = []
            rolling_avg_mse = []
            rolling_avg_rmse = []
            test_scatter_targ = []
            test_scatter_pred = []
            t2m_data_array = []
            for j in range(len(self.months_tar)):
                #month = months_tar[j]
                month = self.cities_data[self.unique_cities[i]][j]
                target_fps_names = []
                target_fps = glob.glob(os.path.join(self.target_file, "*.tif"))
                for x in range(int(len(month))): 
                    value= [".".join(os.path.basename(month[x]).split(".")[0:5])]
                    #print(value)
                    target_fps_names.append(value[0])
                #print("loop count is ----",j,"\n",target_fps_names,"\n")


                #print(target_fps_names)
                predict_fps = [os.path.join(self.predict_file, f"{x}.inputs_pred.tif") for x in target_fps_names]
                target_fps = [glob.glob(os.path.join(self.target_file, f"{x}.*.tif")) for x in target_fps_names]
                input_fps = [glob.glob(os.path.join(self.input_file, f"{x}.inputs.tif")) for x in target_fps_names]

                target_fps = []
                target_fps_2 = [glob.glob(os.path.join(self.target_file, f"{x}.*.tif")) for x in target_fps_names]

                for k in range(int(len(month))):
                    target_fps.append(target_fps_2[k][0])

                for tfp, pfp, ifp  in tqdm(zip(target_fps, predict_fps, input_fps), total=len(target_fps)):
                        if not (os.path.isfile(tfp) and os.path.isfile(pfp) and os.path.isfile(ifp[0])):
                            print(f"Skipping:  Either one of these files missing: \n{tfp}\n{pfp}")
                            continue

                        # read in target data
                        test_band =rasterio.open(ifp[0])
                        ds_target = xr.open_dataset(tfp)
                        ds_target = ds_target.where(ds_target['band_data'] != -9999., np.nan)
                        ds_pred = xr.open_dataset(pfp)
                        ds_pred = ds_pred.where(~np.isnan(ds_target['band_data']), np.nan)
                        # calculate MAE
                        test_mae = mae(ds_pred, ds_target)
                        test_mse = self.mse(ds_pred, ds_target)
                        #print(test_mae,"\n")
                        store_mae.append(test_mae)
                        store_mse.append(test_mse)
                        #print(store_mae,"\n")
                        test_scatter_targ.append(np.squeeze(ds_target['band_data'].values))
                        test_scatter_pred.append(np.squeeze(ds_pred['band_data'].values))
                        t2m_data_array.append(np.nanmean(test_band.read(8)))

                avg_mae = np.mean(store_mae)
                #print("This is MAE",avg_mae,"and this is month",j,"for",unique_cities[i],"\n")

                avg_mse = np.mean(store_mse)
                rmse = avg_mse**(1/2)
                monthly_mae.append(avg_mae)
                monthly_mse.append(avg_mse)
                monthly_rmse.append(rmse)

            window_size = 3

            k = 0
            # Initialize an empty list to store moving averages
            # Loop through the array to consider
            # every window of size 3
            while k < len(monthly_mae) - window_size + 1:
                # Store elements from i to i+window_size
                # in list to get the current window
                window_mae = monthly_mae[k : k + window_size]
                window_mse = monthly_mse[k : k + window_size]
                window_rmse = monthly_rmse[k : k + window_size]
                # Calculate the average of current window
                window_average_mae = round(sum(window_mae) / window_size, 2)
                window_average_mse = round(sum(window_mse) / window_size, 2)
                window_average_rmse = round(sum(window_rmse) / window_size, 2)
                # Store the average of current
                # window in moving average list
                rolling_avg_mae.append(window_average_mae)
                rolling_avg_mse.append(window_average_mse)
                rolling_avg_rmse.append(window_average_rmse)
                # Shift window to right by one position
                k += 1
            Nov_Dec_Jan=np.mean([monthly_mae[10],monthly_mae[11],monthly_mae[0]])
            Dec_Jan_Feb=np.mean([monthly_mae[11],monthly_mae[0],monthly_mae[1]])
            rolling_avg_mae.append(Nov_Dec_Jan)
            rolling_avg_mae.append(Dec_Jan_Feb)

            Nov_Dec_Jan=np.mean([monthly_mse[10],monthly_mse[11],monthly_mse[0]])
            Dec_Jan_Feb=np.mean([monthly_mse[11],monthly_mse[0],monthly_mse[1]])
            rolling_avg_mse.append(Nov_Dec_Jan)
            rolling_avg_mse.append(Dec_Jan_Feb)

            Nov_Dec_Jan=np.mean([monthly_rmse[10],monthly_rmse[11],monthly_rmse[0]])
            Dec_Jan_Feb=np.mean([monthly_rmse[11],monthly_rmse[0],monthly_rmse[1]])
            rolling_avg_rmse.append(Nov_Dec_Jan)
            rolling_avg_rmse.append(Dec_Jan_Feb)

            #print("This is the city:",unique_cities[i],"and the MAE","\n",rolling_avg_mae,"\n")
            #print("This is the city:",unique_cities[i],"and the MSE","\n",rolling_avg_mse,"\n")
            #print("This is the city:",unique_cities[i],"and the RMSE","\n",rolling_avg_rmse,"\n")
        
            self.cities_monthly_mae.update({self.unique_cities[i]:monthly_mae})
            self.cities_monthly_mse.update({self.unique_cities[i]:monthly_mse})
            self.cities_monthly_rmse.update({self.unique_cities[i]:monthly_rmse})

            self.cities_rolling_avg_mae.update({self.unique_cities[i]:rolling_avg_mae})
            self.cities_rolling_avg_mse.update({self.unique_cities[i]:rolling_avg_mse})
            self.cities_rolling_avg_rmse.update({self.unique_cities[i]:rolling_avg_rmse})

            self.scatter_data_targ.update({self.unique_cities[i]:np.array(test_scatter_targ).flatten()})
            self.scatter_data_pred.update({self.unique_cities[i]:np.array(test_scatter_pred).flatten()})
            self.t2m_data.update({self.unique_cities[i]:{"Patch_MAE":store_mae,"t2m_spatial_avg":t2m_data_array}})

    def proccess_city_data(self):
        self.unique_cities_list()
        self.monthly_sort()
        self.rolling_window_calculation()
        self.sort_climate_info_plot_data()
        self.sort_t2m_plot_data()

    def rolling_mae_plot(self):
        plt.figure(figsize=(26, 11.75))
        #print(cities_monthly_mae.shape)
        plot_data_rol = pd.DataFrame.from_dict(self.cities_rolling_avg_mae)
        #print(test,"\n")
        #print(test.transpose(),"\n")
        plot_data_rol = plot_data_rol.transpose()
        #test.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        plot_data_rol.columns = ['JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ','DJF']
        #print(test,"\n")
        #print(test.transpose(),"\n")
        data_rol_plot=sns.heatmap(plot_data_rol,cmap="rainbow",linewidths=0.025,annot=True,cbar_kws={'label':f"Mean Absolute Error \N{DEGREE SIGN}C"})
        #mae_plot.figure.colorbar(mae_plot.collections[0]).set_label('Probability Scale')

        #plt.title("Mean Absolute Error (MAE)", fontsize=16)
        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Cities", fontsize=16)
        #plt.set_label('Probability Scale')
        figure = data_rol_plot.get_figure()
        #figure.savefig('mae_plot_rolling_avg.png',bbox_inches='tight', dpi=500)

    def density_scatter_plots(self):
        correlation_dict = {}

        x_bins = 500
        y_bins = 500
        x_min, x_max = 1, 1
        y_min, y_max = 1, 1

        num_rows = 3
        num_cols = 2
        city_index = 0
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 18.75))

        for i in range(num_rows):
            for j in range(num_cols):
                # Pearson correlations
                min_p = np.nanmin(self.scatter_data_pred[self.unique_cities[city_index]])
                #print(min_p,"\n")
                max_p = np.nanmax(self.scatter_data_pred[self.unique_cities[city_index]])
                #print(max_p,"\n")

                gt_min = np.nanmin(self.scatter_data_targ[self.unique_cities[city_index]])
                gt_max = np.nanmax(self.scatter_data_targ[self.unique_cities[city_index]])

                #min and max across predictions and ground truth
                vmin, vmax = np.minimum(gt_min,min_p), np.maximum(gt_max, max_p)
                #print(vmin," ",vmax)
                #print(LogNorm(vmin=vmin*-1, vmax=vmax))
                #Flatten the target and prediction dtatsets into arrays
                target_line = self.scatter_data_targ[self.unique_cities[city_index]]
                #print(target_line,"\n")
                pred_line = self.scatter_data_pred[self.unique_cities[city_index]]

                #corr_gfm, _ = pearsonr(target_line, pred_line)
                #print(corr_gfm)

                df = pd.DataFrame({"Target":target_line,"Prediction":pred_line})
                df =df.dropna()
                #print(df)
                correlation_value = df['Target'].corr(df['Prediction'])
                #print(correlation_value)
                correlation_dict.update({self.unique_cities[city_index]:[correlation_value]})
                

                # --- GridFM Mass Correlation ---
                h1 = axes[i,j].hist2d(
                    df['Target'], df['Prediction'],
                    bins=[x_bins, y_bins],
                    norm=LogNorm(vmin=abs(vmin), vmax=vmax),
                    cmap="inferno"
                )
                
                axes[i,j].set_xlabel(r'LST Obs [$\degree$ C]', fontsize=12)
                axes[i,j].set_ylabel(r'LST Pred [$\degree$ C]', fontsize=12)
                axes[i,j].plot([vmin, vmax], [vmin, vmax], 'b--')
                axes[i,j].set_title(self.unique_cities[city_index])
                axes[i,j].text(0.3, 0.01, f"Correlation  = {correlation_value:.5f}", transform=axes[i,j].transAxes, fontsize=10, weight='bold')

                # Colorbar
                cbar = fig.colorbar(h1[3], ax=axes[i,j], pad=0.02)
                cbar.set_label("Number of samples", fontsize=10)

                # Style adjustments
                
                axes[i,j].grid(True, linewidth=0.3)
                axes[i,j].tick_params(axis='both', labelsize=10)
                city_index+=1
                #ax1.label_outer(f"Correlation = {correlation_value:.4f}")
                
                #ax1.bar_label(f"Correlation = {correlation_value:.4f}")
        corr_dict =pd.DataFrame.from_dict(correlation_dict).transpose()
        corr_dict.columns = ["Correlation"]
        sorted_corr_dict =corr_dict.sort_values(by='Correlation')
        print(sorted_corr_dict)

        plt.tight_layout()
        #plt.savefig("density_function_scatter_plot.png", bbox_inches="tight")
        plt.show()

    def getClimate_zones_test(self):
        city_info  = pd.read_csv('../assets/databases/global_cities_database.csv',index_col=19)
        city_names = self.unique_cities
        self.climate_zones = {}
        print(city_names[0][:].split("_"))
        for i in tqdm(range(len(city_names))):
            name = city_names[i][:].split("_")
            print(name,"\n")
            country_name = city_info.loc[city_names[i][:],'COUNTRY_NAME']
            print("Country name:",country_name,"\n")
            print("City name:",name[0],"\n")
            # Initialize the Nominatim geocoder
            geolocator = Nominatim(user_agent="my_city_converter")
            # City name to convert
            city_name = [name[0],country_name]
            #print(city_name,"\n")

            # Geocode the city name
            location = geolocator.geocode(city_name)
            #print("This is the city name",city_name[i] )

            # Extract and print latitude and longitude if found
            if location:
                latitude = location.latitude
                longitude = location.longitude
                print(f"The latitude of {city_name} is: {latitude}")
                print(f"The longitude of {city_name} is: {longitude}")
                #print("\n")
            else:
                print(f"Could not find coordinates for {city_name}")

            kg_zone = lookupCZ(latitude, longitude)
            print('Koppen geiger zone is '+ kg_zone,"\n")

            self.climate_zones.update({city_names[i][:]:{"City_Name":name[0],"Climate":kg_zone}})

        #return climate_zones

    def sort_climate_info_plot_data(self):
        self.getClimate_zones_test()
        self.climate_info=pd.DataFrame.from_dict(self.climate_zones).transpose()
        city_mae={}
        self.df_cities_monthly_mae = pd.DataFrame.from_dict(self.cities_monthly_mae)
        for i in range(len(self.df_cities_monthly_mae.columns)):
            city_mae.update({self.climate_info.index[i]:self.df_cities_monthly_mae[self.climate_info.index[i]].mean()})
        city_mae=pd.DataFrame(city_mae,index=[0]).transpose()
        city_mae.columns = ["MAE"]
        self.full_climate_info= self.climate_info.join(city_mae)
        unit = "C"
        self.full_climate_info = self.full_climate_info.rename(columns={'MAE': f"Mean Absolute Error °{unit}"})

        city_samples = defaultdict(int)
        input_files = glob.glob(os.path.join(self.input_file, "*.tif"))
        for i in range(len(input_files)):
            city_samples[input_files[i].split("/")[-1].split(".")[0]] +=1

        self.full_climate_info["Major_KCC"] = None
        self.full_climate_info["Num_of_input_samples"] = None
        print(len(self.full_climate_info.index))
        major_kcc = ['A(tropical)','B(arid)','C(temperate)','D(continental)','E(polar)']

        for i in range(len(self.full_climate_info.index)):
            self.full_climate_info.loc[self.full_climate_info.index[i],'Num_of_input_samples'] = int(city_samples[self.full_climate_info.index[i]])
            self.full_climate_info.loc[self.full_climate_info.index[i],'City_tag'] = str(i)+"_"+self.full_climate_info.loc[self.full_climate_info.index[i]]['Climate'][0]
            
            if self.full_climate_info.loc[self.full_climate_info.index[i]]['Climate'][0] == 'A':
                self.full_climate_info.loc[self.full_climate_info.index[i],'Major_KCC'] = major_kcc[0]
            elif self.full_climate_info.loc[self.full_climate_info.index[i]]['Climate'][0] == 'B':
                self.full_climate_info.loc[self.full_climate_info.index[i],'Major_KCC'] = major_kcc[1]
            elif self.full_climate_info.loc[self.full_climate_info.index[i]]['Climate'][0] == 'C':
                self.full_climate_info.loc[self.full_climate_info.index[i],'Major_KCC'] = major_kcc[2]
            elif self.full_climate_info.loc[self.full_climate_info.index[i]]['Climate'][0] == 'D':
                self.full_climate_info.loc[self.full_climate_info.index[i],'Major_KCC'] = major_kcc[3]
            elif self.full_climate_info.loc[self.full_climate_info.index[i]]['Climate'][0] == 'E':
                self.full_climate_info.loc[self.full_climate_info.index[i],'Major_KCC'] = major_kcc[4]
            else:
                print("No major koppen climate classification found for the city")

   
    def first_climate_info_plot(self):
        grouped = self.full_climate_info.groupby("City_Name")["Mean Absolute Error °C"].mean().sort_values(ascending=False).index
        plt.figure(figsize=(15, 12))
        ax = sns.barplot(data=self.full_climate_info,x="City_Name", y='Mean Absolute Error °C', hue="Climate",order=grouped)
        ax.tick_params(axis='x', labelrotation=90)
        plt.tight_layout()
        plt.show()

    def second_climate_info_plot(self):
        grouped = self.full_climate_info.groupby(["Climate","City_Name"])['Mean Absolute Error °C'].sum().reset_index()
        plt.figure(figsize=(15, 12))
        ax = sns.barplot(data=self.full_climate_info,x="City_Name", y='Mean Absolute Error °C', hue="Climate",order=grouped)
        ax.tick_params(axis='x', labelrotation=90)
        plt.tight_layout()
        plt.show()

    def third_climate_info_plot(self):
        plt.figure(figsize=(15, 12))
        ax = sns.barplot(data=self.full_climate_info,x="City_Name", y='Mean Absolute Error °C', hue="Major_KCC")
        ax.tick_params(axis='x', labelrotation=90)
        plt.tight_layout()
        plt.show()

    def forth_climate_info_plot(self):
        sample_data = self.full_climate_info.groupby('Major_KCC').agg(
        Mean_Absolute_Error = ('Mean Absolute Error °C','mean'),
        Num_of_input_samples =  ('Num_of_input_samples', 'sum'))
        print(sample_data)
        plt.figure(figsize=(15, 12))
        ax = sns.barplot(data=sample_data,x="Num_of_input_samples", y='Mean_Absolute_Error', hue="Major_KCC")
        ax.tick_params(axis='x', labelrotation=0)
        plt.tight_layout()
        plt.show()
        

    def sort_t2m_plot_data(self):
        self.df_all_cites = pd.DataFrame(columns=['City_Name','Patch_MAE','t2m_spatial_avg','Climate'])
        data = 0
        for i in tqdm(range(len(self.climate_info.index))):
            #for j in range(len(self.t2m_data[self.climate_info.index[i]]['t2m_spatial_avg'].strip('[]').split(","))):
            for j in range(len(self.t2m_data[self.climate_info.index[i]]['t2m_spatial_avg'])):
                data =self.climate_info.loc[[self.climate_info.index[i]]]
                #print(data)
                new_data = pd.DataFrame({'City_Name':[data['City_Name'][0]],
                                    'Patch_MAE':[self.t2m_data[self.climate_info.index[i]]['Patch_MAE'][j]],
                                    't2m_spatial_avg':[self.t2m_data[self.climate_info.index[i]]['t2m_spatial_avg'][j]],
                                    'Climate':[data['Climate'][0]]})
                self.df_all_cites = pd.concat([self.df_all_cites, new_data])
        self.df_all_cites['Patch_MAE'] = self.df_all_cites['Patch_MAE'].astype(float)
        self.df_all_cites['t2m_spatial_avg'] = self.df_all_cites['t2m_spatial_avg'].astype(float)
        print(self.df_all_cites)

    def t2m_scatter_plot(self):
        plt.figure(figsize=(15, 12))
        sns.scatterplot(data=self.df_all_cites, x="t2m_spatial_avg", y="Patch_MAE",hue="Climate",palette="deep")

    def t2m_violin_plot(self):
        plt.figure(figsize=(15, 12))
        sns.violinplot(data=self.df_all_cites, x='Climate', y='t2m_spatial_avg',hue="Climate",palette="deep")

    def t2m_patch_mae(self):
        plt.figure(figsize=(15, 12))
        sns.violinplot(data=self.df_all_cites, x='Climate', y='Patch_MAE',hue="Climate",palette="deep")

    def t2m_pair_plot(self):
        sns.pairplot(self.df_all_cites, hue="Climate", height=7.5)

    def t2m_joint_plot(self):
        sns.jointplot(x='t2m_spatial_avg', y='Patch_MAE', data=self.df_all_cites,hue="Climate",height=10)
        plt.suptitle("Joint Density Plot", y=1.02) # Adjust suptitle position
        plt.show()