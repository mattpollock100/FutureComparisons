# Import various python packages

import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.coord_categorisation as icc
from iris.time import PartialDateTime
import iris.analysis

import xarray as xr

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pandas as pd

import os
import warnings 
import io

import cartopy.io.shapereader as shpreader


from matplotlib.path import Path
import matplotlib.patches as mpatches
import calendar
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import get_cmap
from matplotlib.legend_handler import HandlerBase

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from scipy.optimize import minimize
from scipy.stats import norm

from scipy.interpolate import PchipInterpolator
from scipy.optimize import differential_evolution

import operator

import pickle
from copy import deepcopy
from pathlib import Path

import csv

import dask
from dask.diagnostics import ProgressBar


warnings.filterwarnings('ignore') 

import sys
sys.path.append('..')

from Cube_Functions import *
from Plot_Functions import *
from Climate_Functions import *
from Stats_Functions import *
print('Finished loading libraries and functions')

#clean up namespace
del sys

#Primary settings 

pickle_save = False
pickle_load = False



models = [
            'ACCESS-ESM1-5',
            'CESM2',
            'IPSL-CM6A-LR',
            'MIROC-ES2L',
            'NESM3',
            ]

models = ['ACCESS-ESM1-5',   #For testing
            'NESM3']



control = 'piControl'
transient = '1pctCO2'
paleo = 'lig127k' #Analysis is run on paleo
paleo2 = 'midHolocene' #set paleo2 to None to not use it, or set it to a string to use it (e.g. 'midHolocene'). Analysis still runs on paleo, but paleo2 data is loaded for comparison and plotting.



root_path = '/gws/nopw/j04/pmip4_vol1/public/matt/data/'

energy_var_dict = {
    'rsds': 1,
    'rsus': -1,
    'rlds': 1,
    'rlus': -1,
    'rsdscs': 1,
    'rldscs': 1,
    'hfls': -1,
    'hfss': -1
}

cloud_vars = ['rsdscs', 'rldscs']

energy_vars = list(energy_var_dict.keys())
ice_vars = ['siconc', 'sithick']
other_vars = ['tos', 'clt']


var_list = ice_vars + energy_vars + other_vars

sia_crossing = None   #Set to a number to use an absolute value for the crossing point, or set to None to use the crossing point determined by the paleo siconc time series.

if sia_crossing is not None:
    ts_to_use = 'ts'
else:
    ts_to_use = 'anom_ts'

season = 'Summer'

if season == 'Winter':
    paleo_month = 'Mar'
    paleo2_month = 'Mar'
    transient_month = 'Mar'
    control_month = 'Mar'
elif season == 'Summer':
    paleo_month = 'Aug'
    paleo2_month = 'Sep'
    transient_month = 'Sep'
    control_month = 'Sep'

# Climatology plot settings
x = np.array([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345])
m = np.arange(12)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

paper_colours = {
    "lig127k":              "#077907",
    "lig127k Crossing":           "#2EDF8C",
    "midHolocene":              "#AA3377",
    "midHolocene Crossing":           "#EE6677",
    "piControl": "black",
    "total_energy": "#D62E33",  # muted red
    "cs_downwelling_energy": "#D62E33",  # muted red
    "sigrowth":         "#009E73",  # bluish green
    "tos":                    "#0072B2",  # strong blue
}

plot_path = '/home/users/matt/FutureComparisons/Figures/'


#Define region

con = iris.Constraint(latitude=lambda lat: 60 <= lat <= 90)

time_con = iris.Constraint(year=lambda y:np.logical_and(y>=2035,y<2045))  #time_constraint. Not used anywhere yet.

ocean_shp_reader = shpreader.Reader(shpreader.natural_earth(resolution="110m", category="physical", name="ocean"))
ocean_list = []
for ocean in ocean_shp_reader.records():
    ocean_list.append(ocean.geometry)
ocean_shp = ocean_list[1]

shape = ocean_shp

#some data stored as cubes, some as arrays. This clears it up
def as_array(x):
    if isinstance(x, iris.cube.Cube):
        return x.data
    return np.asarray(x)

def keep_full_years(cube, time_coord_name='time'):
    """
    Return a cube containing only complete calendar years:
    exactly one of each month Jan..Dec.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube with a time coordinate.
    time_coord_name : str, optional
        Name of the time coordinate.

    Returns
    -------
    iris.cube.Cube
        Cube containing only full years.
    """
    time_coord = cube.coord(time_coord_name)
    datetimes = time_coord.units.num2date(time_coord.points)

    years = np.array([dt.year for dt in datetimes])
    months = np.array([dt.month for dt in datetimes])

    full_years = []

    for year in np.unique(years):
        year_months = months[years == year]

        # Require exactly 12 points and exactly months 1..12
        if len(year_months) == 12 and np.array_equal(np.sort(year_months), np.arange(1, 13)):
            full_years.append(year)

    full_years = np.array(full_years)

    if len(full_years) == 0:
        raise ValueError("No complete years found in cube.")

    mask = np.isin(years, full_years)

    return cube[mask]

def realise_cube(cube, scheduler="single-threaded", show_progress=True):
    """
    Return a copy of an Iris cube with realised, non-lazy data.

    This prevents small-looking derived cubes from retaining large Dask graphs
    back to the original source data.
    """
    if not cube.has_lazy_data():
        return cube.copy()

    arr = cube.core_data()

    with dask.config.set(scheduler=scheduler):
        if show_progress:
            with ProgressBar():
                data = arr.compute()
        else:
            data = arr.compute()

    return cube.copy(data=data)


def get_cube_dict(model, experiment, var_list, root_path, con, shape):
    cube_dict = {}
    print(f'Loading cubes', end='...')
    for var in var_list:
        print(f'{var}', end='...')
        path = create_path(model, experiment, var, root_path, regrid_label='cdo')
        cube = get_cube(path, var, con, shape=shape)
        #need to trim incomplete years from start and end
        #trimmed_cube = keep_full_years(cube)
        #cube_dict[var] = trimmed_cube
        cube_dict[var] = cube
    print('Done.')
    return cube_dict


# def get_ts_dict_from_cube_dict(cube_dict, var_list, energy_var_dict):
#     ts_dict = {}
#     energy_vars = list(energy_var_dict.keys())
#     print(f'Calculating time series for', end='...')
#     for var in var_list:
#         print(f'{var}', end='...')


#         cube = cube_dict[var]
#         if var == 'siconc' or var == 'clt':
#             analysis = 'sum'
#         else:        
#             analysis = 'mean'
#         if var in energy_vars:
#             multiplier = energy_var_dict[var]
#         else:
#             multiplier = 1

#         area_weights = iris.analysis.cartography.area_weights(cube)
#         op = get_iris_op(analysis)

#         timeseries = (cube.collapsed(['latitude', 'longitude'], op, weights=area_weights)) * multiplier

#         ts_dict[var] = timeseries

#     print('Done.')
#     return ts_dict


def get_ts_dict_from_cube_dict(
    cube_dict,
    var_list,
    energy_var_dict,
    realise=True,
    scheduler="single-threaded",
    show_progress=True,
):
    ts_dict = {}
    energy_vars = list(energy_var_dict.keys())

    print("Calculating time series for", end="...")

    for var in var_list:
        print(f"{var}", end="...")

        cube = cube_dict[var]

        if var == "siconc" or var == "clt":
            analysis = "sum"
        else:
            analysis = "mean"

        if var in energy_vars:
            multiplier = energy_var_dict[var]
        else:
            multiplier = 1

        area_weights = iris.analysis.cartography.area_weights(cube)
        op = get_iris_op(analysis)

        timeseries = cube.collapsed(
            ["latitude", "longitude"],
            op,
            weights=area_weights,
        )

        timeseries = timeseries * multiplier

        if realise:
            print("realising", end="...")
            timeseries = realise_cube(
                timeseries,
                scheduler=scheduler,
                show_progress=show_progress,
            )

        ts_dict[var] = timeseries

    print("Done.")
    return ts_dict

# def get_clim_dict_from_ts_dict(ts_dict):

#     var_list = list(ts_dict.keys())

#     clim_dict = {}
#     print(f'Calculating climatology for',end="...")
#     for var in var_list:
#         print(f'{var}',end="...")

#         timeseries = ts_dict[var]

#         analysis = 'mean'

#         op = get_iris_op(analysis)

#         clim = timeseries.aggregated_by('month', op)

#         clim_dict[var] = clim

#     print('Done.')
#     return clim_dict

def get_clim_dict_from_ts_dict(
    ts_dict,
    realise=True,
    scheduler="single-threaded",
    show_progress=True,
):
    var_list = list(ts_dict.keys())

    clim_dict = {}

    print("Calculating climatology for", end="...")

    for var in var_list:
        print(f"{var}", end="...")

        timeseries = ts_dict[var]

        op = get_iris_op("mean")

        clim = timeseries.aggregated_by("month", op)

        if realise:
            print("realising", end="...")
            clim = realise_cube(
                clim,
                scheduler=scheduler,
                show_progress=show_progress,
            )

        clim_dict[var] = clim

    print("Done.")
    return clim_dict

def calculate_anomalies(experiment_ts_dict, control_clim_dict = None):    
    experiment_anom_ts_dict = {}
    print(f'Calculating anomalies for', end='...')
    for var, ts in experiment_ts_dict.items():
        print(f'{var}',end="...")
        if control_clim_dict is None:
            #take first 10 years as control climatology
            clim = ts[:120].aggregated_by('month', iris.analysis.MEAN)
        else:
            clim = control_clim_dict[var]
        clim_data = clim.data
        ts_months = ts.coord("month_number").points
        
        month_index = ts_months.astype(int) - 1

        anomalies = ts.copy()
        anomalies.data = ts.data - clim_data[month_index]

        experiment_anom_ts_dict[var] = anomalies
    print('Done.')
    return experiment_anom_ts_dict

#Create dictionaries from data stored in the netcdf files
if not pickle_load:
  control_big_dict = {}
  paleo_big_dict = {}

  transient_big_dict = {}

  if paleo2:
    paleo2_big_dict = {}

  for model in models:
    print(f'------------------Processing {model} {control}------------------')
    control_cube_dict = get_cube_dict(model, control, var_list, root_path, con, shape)
    control_ts_dict = get_ts_dict_from_cube_dict(control_cube_dict, var_list, energy_var_dict)
    control_clim_dict = get_clim_dict_from_ts_dict(control_ts_dict)

    control_big_dict[model] = {'cubes': control_cube_dict, 'ts': control_ts_dict, 'clim': control_clim_dict}

    print(f'------------------Processing {model} {transient}------------------')
    transient_cube_dict = get_cube_dict(model, transient, var_list, root_path, con, shape)
    transient_ts_dict = get_ts_dict_from_cube_dict(transient_cube_dict, var_list, energy_var_dict)
    transient_clim_dict = get_clim_dict_from_ts_dict(transient_ts_dict)


    transient_anom_ts_dict = calculate_anomalies(transient_ts_dict , control_clim_dict) 
        
    transient_anom_clim_dict = get_clim_dict_from_ts_dict(transient_anom_ts_dict)

    transient_big_dict[model] = {'cubes': transient_cube_dict, 
                                'ts': transient_ts_dict, 
                                'clim': transient_clim_dict,
                                'anom_ts': transient_anom_ts_dict,
                                'anom_clim': transient_anom_clim_dict,
                                }
    
    

    print(f'------------------Processing {model} {paleo}------------------')
    paleo_cube_dict = get_cube_dict(model, paleo, var_list, root_path, con, shape)
    paleo_ts_dict = get_ts_dict_from_cube_dict(paleo_cube_dict, var_list, energy_var_dict)
    paleo_clim_dict = get_clim_dict_from_ts_dict(paleo_ts_dict)
    paleo_anom_ts_dict = calculate_anomalies(paleo_ts_dict, control_clim_dict)
    paleo_anom_clim_dict = get_clim_dict_from_ts_dict(paleo_anom_ts_dict)

    paleo_big_dict[model] = {'cubes': paleo_cube_dict, 
                            'ts': paleo_ts_dict, 
                            'clim': paleo_clim_dict,
                            'anom_ts': paleo_anom_ts_dict,
                            'anom_clim': paleo_anom_clim_dict,
                            }
    if paleo2:
      print(f'------------------Processing {model} {paleo2}------------------')
      paleo2_cube_dict = get_cube_dict(model, paleo2, var_list, root_path, con, shape)
      paleo2_ts_dict = get_ts_dict_from_cube_dict(paleo2_cube_dict, var_list, energy_var_dict)
      paleo2_clim_dict = get_clim_dict_from_ts_dict(paleo2_ts_dict)
      paleo2_anom_ts_dict = calculate_anomalies(paleo2_ts_dict, control_clim_dict)
      paleo2_anom_clim_dict = get_clim_dict_from_ts_dict(paleo2_anom_ts_dict)

      paleo2_big_dict[model] = {'cubes': paleo2_cube_dict, 
                                'ts': paleo2_ts_dict, 
                                'clim': paleo2_clim_dict,
                                'anom_ts': paleo2_anom_ts_dict,
                                'anom_clim': paleo2_anom_clim_dict,
                                }


#create the same dictionaries based on the crossing point, where transient and paleo runs have equal SIA
def get_crossing_big_dict(models, paleo_big_dict, transient_big_dict, sia_crossing=None, window=10):
    

    crossing_big_dict = {}
    for model in models:
        
        if sia_crossing is None:
            paleo_siconc_anom = paleo_big_dict[model][ts_to_use]['siconc'].extract(iris.Constraint(month=paleo_month)).collapsed('year', iris.analysis.MEAN).data / 1e14
        else:
            paleo_siconc_anom = sia_crossing
            
        transient_siconc_ts_anom = transient_big_dict[model][ts_to_use]['siconc'].extract(iris.Constraint(month=transient_month)) / 1e14
        
        transient_min =  transient_siconc_ts_anom.data

        transient_min_rolling = pd.Series(transient_min).rolling(window=window, center=True).mean()

        n_years = transient_min_rolling.shape[0]
        years = np.arange(n_years) + 1

        crossing_mask = transient_min_rolling.to_numpy() <= paleo_siconc_anom

        if not np.any(crossing_mask):
            print(f"{model}: no crossing found")
            continue

        sia_crossing_idx = np.where(crossing_mask)[0][0]
        sia_crossing_year = years[sia_crossing_idx]
        #sia_crossing_year = years[transient_min_rolling <= paleo_siconc_anom][0]

        print(f'{model} paleo Aug siconc anomaly: {paleo_siconc_anom:.2f}, transient crossing year: {sia_crossing_year}')


        transient_siconc_ts_anom = (
        transient_big_dict[model][ts_to_use]['siconc']
        .extract(iris.Constraint(month=transient_month))
        / 1e14
        )

        transient_min = transient_siconc_ts_anom.data
        years_coord = transient_siconc_ts_anom.coord('year').points

        transient_min_rolling = (
            pd.Series(transient_min, index=years_coord)
            .rolling(window=window, center=True)
            .mean()
        )

        crossing_mask = transient_min_rolling.to_numpy() <= paleo_siconc_anom

        if not np.any(crossing_mask):
            print(f"{model}: no crossing found")
            continue

        crossing_year_value = transient_min_rolling.index[np.where(crossing_mask)[0][0]]
        
        crossing_cubes_dict = {}
        crossing_ts_dict = {}
        crossing_clim_dict = {}
        crossing_anom_ts_dict = {}
        crossing_anom_clim_dict = {}
        
        for var in var_list:

            # unique_years = transient_big_dict[model][ts_to_use]['siconc'].coord('year').points
            # crossing_year_value = unique_years[sia_crossing_year - 1]

            half_window = window // 2
            year_constraint = iris.Constraint(
                year=lambda y: crossing_year_value - half_window <= y < crossing_year_value + half_window
                )

            cube = transient_big_dict[model]['cubes'][var].extract(year_constraint)
            crossing_cubes_dict[var] = cube

            ts = transient_big_dict[model]['ts'][var].extract(year_constraint)
            crossing_ts_dict[var] = ts

            anom_ts = transient_big_dict[model]['anom_ts'][var].extract(year_constraint)
            crossing_anom_ts_dict[var] = anom_ts


        crossing_clim_dict = get_clim_dict_from_ts_dict(crossing_ts_dict)
        crossing_anom_clim_dict = get_clim_dict_from_ts_dict(crossing_anom_ts_dict)
        
        crossing_big_dict[model] = {'cubes': crossing_cubes_dict,
                                'ts': crossing_ts_dict,
                                'clim': crossing_clim_dict,
                                'anom_ts': crossing_anom_ts_dict,
                                'anom_clim': crossing_anom_clim_dict,
                                }
    return crossing_big_dict



if not pickle_load: crossing_big_dict = get_crossing_big_dict(models, paleo_big_dict, transient_big_dict, sia_crossing=sia_crossing, window=10)

if not pickle_load: crossing2_big_dict = get_crossing_big_dict(models, paleo2_big_dict, transient_big_dict, sia_crossing=sia_crossing, window=10)

#add cloud effect to the dictionaries
if not pickle_load:
    for model in models:
        for experiment, big_dict in zip([control, paleo, f'{paleo} Crossing', paleo2, f'{paleo2} Crossing2'], [control_big_dict, paleo_big_dict, crossing_big_dict, paleo2_big_dict, crossing2_big_dict]):
            if paleo2 is None and paleo2 in experiment:
                continue
            print(f'Adding cloud effect to {model} {experiment} big dict')
            sw_clim = big_dict[model]['clim']['rsds']
            lw_clim = big_dict[model]['clim']['rlds']
            swcs_clim = big_dict[model]['clim']['rsdscs']
            lwcs_clim = big_dict[model]['clim']['rldscs']
            
            sw_cloudeffect = (sw_clim - swcs_clim) / swcs_clim
            lw_cloudeffect = (lw_clim - lwcs_clim) / lwcs_clim
            total_cloudeffect = (sw_clim - swcs_clim + lw_clim - lwcs_clim) #/ (swcs_clim + lwcs_clim) #Don't divide for total, keep as Wm-2
            big_dict[model]['clim']['sw_cloudeffect'] = sw_cloudeffect
            big_dict[model]['clim']['lw_cloudeffect'] = lw_cloudeffect
            big_dict[model]['clim']['total_cloudeffect'] = total_cloudeffect
            if experiment == control:
                control_sw_cloudeffect = sw_cloudeffect.copy()
                control_lw_cloudeffect = lw_cloudeffect.copy()
                control_total_cloudeffect = total_cloudeffect.copy()
            else:
                sw_cloudeffect_anom = sw_cloudeffect.data - control_sw_cloudeffect.data
                lw_cloudeffect_anom = lw_cloudeffect.data - control_lw_cloudeffect.data
                total_cloudeffect_anom = total_cloudeffect.data - control_total_cloudeffect.data
                big_dict[model]['anom_clim']['sw_cloudeffect'] = sw_cloudeffect_anom
                big_dict[model]['anom_clim']['lw_cloudeffect'] = lw_cloudeffect_anom
                big_dict[model]['anom_clim']['total_cloudeffect'] = total_cloudeffect_anom

#Add growth estimates to the dictionaries, based on volumes

def compute_monthly_growth_from_volumes_ts(volume_ts):
    """
    Estimate calendar-month SIV growth from monthly-mean SIV time series.

    growth[t] ≈ end-of-month volume - start-of-month volume.

    Uses adjacent monthly means to estimate month-boundary volumes.
    First and last months are masked because their outer boundaries
    cannot be estimated without data outside the time series.
    """
    volume_ts = np.ma.asarray(volume_ts)

    growth = np.ma.masked_all_like(volume_ts)

    # For month t:
    # start_t ≈ 0.5 * (V[t-1] + V[t])
    # end_t   ≈ 0.5 * (V[t] + V[t+1])
    # growth_t = end_t - start_t
    growth[1:-1] = 0.5 * (volume_ts[2:] - volume_ts[:-2])

    return growth


def compute_monthly_growth_from_volumes_clim(volume_clim, month_lengths_days=None):
    """
    Estimate calendar-month volume growth from 12 monthly mean values.

    Assumes monthly means represent values at month centres.
    Interpolates a smooth periodic annual cycle and evaluates volume at
    month boundaries.

    Returns
    -------
    growth : np.ndarray
        Monthly growth, Jan-Dec.
        Positive = volume increase.
        Negative = volume loss.
    """

    volume_clim = np.asarray(volume_clim, dtype=float)

    if volume_clim.shape[0] != 12:
        raise ValueError("Expected 12 monthly climatological values")

    if month_lengths_days is None:
        month_lengths_days = np.array([
            31, 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31
        ], dtype=float)

    year_length = month_lengths_days.sum()

    # Month boundaries: start Jan = 0, start Feb = 31, etc.
    month_bounds = np.concatenate([[0], np.cumsum(month_lengths_days)])

    # Month centres
    month_centres = month_bounds[:-1] + 0.5 * month_lengths_days

    # Periodic extension: previous year, current year, next year
    t_ext = np.concatenate([
        month_centres - year_length,
        month_centres,
        month_centres + year_length
    ])

    v_ext = np.concatenate([
        volume_clim,
        volume_clim,
        volume_clim
    ])

    # Shape-preserving interpolation
    interp = PchipInterpolator(t_ext, v_ext, extrapolate=False)

    start_volumes = interp(month_bounds[:-1])
    end_volumes = interp(month_bounds[1:])

    growth = end_volumes - start_volumes

    return growth



#add volume cubes to big dicts
if not pickle_load:
    for model in models:
        for experiment, big_dict in zip([control, transient, paleo, paleo2, f'{paleo} Crossing', f'{paleo2} Crossing2'], [control_big_dict, transient_big_dict, paleo_big_dict, paleo2_big_dict, crossing_big_dict, crossing2_big_dict]):
            if paleo2 is None and experiment == paleo2:
                continue
            print(f'Adding volume cubes to {model} {experiment} big dict')
            cube_dict = big_dict[model]['cubes']
            siconc_cube = cube_dict['siconc']
            sithick_cube = cube_dict['sithick']
            siconc_cube_overlap, sithick_cube_overlap = extract_time_overlap(siconc_cube, sithick_cube)
            volume_cube = (siconc_cube_overlap * sithick_cube_overlap) / 100 #divide by 100 as siconc is in percentage, and we want it as a fraction for volume calculation

            big_dict[model]['cubes']['sivol'] = volume_cube

            area = iris.analysis.cartography.area_weights(volume_cube)

            ts = volume_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=area)
            big_dict[model]['ts']['sivol'] = ts

            clim = big_dict[model]['ts']['sivol'].aggregated_by('month', iris.analysis.MEAN)
            big_dict[model]['clim']['sivol'] = clim

            sigrowth_ts = compute_monthly_growth_from_volumes_ts(ts.data)
            sigrowth_ts_cube = ts.copy()
            sigrowth_ts_cube.data = sigrowth_ts
            big_dict[model]['ts']['sigrowth'] = sigrowth_ts_cube

            sigrowth_clim = compute_monthly_growth_from_volumes_clim(clim.data)
            sigrowth_clim_cube = clim.copy()
            sigrowth_clim_cube.data = sigrowth_clim
            big_dict[model]['clim']['sigrowth'] = sigrowth_clim_cube

            if experiment == control:
                control_clim = clim.copy()
                control_growth_clim = sigrowth_clim.copy()
            else:
                ts_months = ts.coord("month_number").points
                month_index = ts_months.astype(int) - 1

                anom_ts = ts.copy()
                anom_ts.data = ts.data - control_clim.data[month_index]
                big_dict[model]['anom_ts']['sivol'] = anom_ts

                anom_clim = clim.copy()
                anom_clim.data = clim.data - control_clim.data
                big_dict[model]['anom_clim']['sivol'] = anom_clim

                anom_ts_growth = sigrowth_ts_cube.copy()
                anom_ts_growth.data = sigrowth_ts - control_growth_clim[month_index]
                big_dict[model]['anom_ts']['sigrowth'] = anom_ts_growth

                anom_growth_clim = sigrowth_clim_cube.copy()
                anom_growth_clim.data = sigrowth_clim - control_growth_clim
                big_dict[model]['anom_clim']['sigrowth'] = anom_growth_clim


if not pickle_load:
    for big_dict in [paleo_big_dict, crossing_big_dict, paleo2_big_dict, crossing2_big_dict]:
        for model in models:
            #add area to big dicts
            cube_dict = big_dict[model]['cubes']
            siconc_cube = cube_dict['siconc'][0]
            area_weights = iris.analysis.cartography.area_weights(siconc_cube)
            masked_weights = np.ma.array(area_weights, mask=siconc_cube.data.mask)
            total_area_m2 = masked_weights.sum()
            big_dict[model]['area'] = total_area_m2
            print(f'{model} area: {total_area_m2:.2e} m²')

#create total energy budget variable and add to big dicts
if not pickle_load:
    for big_dict in [control_big_dict, transient_big_dict, paleo_big_dict, paleo2_big_dict, crossing_big_dict, crossing2_big_dict]:
        for model in models:
            print(f'Calculating total energy budget for {model} and adding to big dict')
            cube_dict = big_dict[model]['cubes']
            rsds_cube = cube_dict['rsds']
            rsus_cube = cube_dict['rsus']
            rlds_cube = cube_dict['rlds']
            rlus_cube = cube_dict['rlus']
            hfls_cube = cube_dict['hfls']
            hfss_cube = cube_dict['hfss']


            total_energy_cube = (rsds_cube - rsus_cube) + (rlds_cube - rlus_cube) - hfls_cube - hfss_cube
            downwelling_energy = rsds_cube + rlds_cube

            big_dict[model]['cubes']['total_energy'] = total_energy_cube
            big_dict[model]['cubes']['downwelling_energy'] = downwelling_energy

            area_weights = iris.analysis.cartography.area_weights(total_energy_cube)
            total_energy_ts = total_energy_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=area_weights)
            big_dict[model]['ts']['total_energy'] = total_energy_ts

            area_weights = iris.analysis.cartography.area_weights(downwelling_energy)
            downwelling_energy_ts = downwelling_energy.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=area_weights)
            big_dict[model]['ts']['downwelling_energy'] = downwelling_energy_ts

            total_energy_clim = total_energy_ts.aggregated_by('month', iris.analysis.MEAN)
            big_dict[model]['clim']['total_energy'] = total_energy_clim

            
            downwelling_energy_clim = downwelling_energy_ts.aggregated_by('month', iris.analysis.MEAN)
            big_dict[model]['clim']['downwelling_energy'] = downwelling_energy_clim

            if 'anom_clim' in big_dict[model].keys():
                control_total_energy_clim = control_big_dict[model]['clim']['total_energy']
                total_energy_anom_clim = total_energy_clim.copy()
                total_energy_anom_clim.data = total_energy_clim.data - control_total_energy_clim.data
                big_dict[model]['anom_clim']['total_energy'] = total_energy_anom_clim

                control_total_energy_ts = control_big_dict[model]['ts']['total_energy']
                total_energy_anom_ts = total_energy_ts.copy()
                ts_months = total_energy_ts.coord("month_number").points
                month_index = ts_months.astype(int) - 1
                total_energy_anom_ts.data = total_energy_ts.data - control_total_energy_ts.data[month_index]
                big_dict[model]['anom_ts']['total_energy'] = total_energy_anom_ts

                control_downwelling_energy_clim = control_big_dict[model]['clim']['downwelling_energy']
                downwelling_energy_anom_clim = downwelling_energy_clim.copy()
                downwelling_energy_anom_clim.data = downwelling_energy_clim.data - control_downwelling_energy_clim.data
                big_dict[model]['anom_clim']['downwelling_energy'] = downwelling_energy_anom_clim

                control_downwelling_energy_ts = control_big_dict[model]['ts']['downwelling_energy']
                downwelling_energy_anom_ts = downwelling_energy_ts.copy()
                ts_months = downwelling_energy_ts.coord("month_number").points
                month_index = ts_months.astype(int) - 1
                downwelling_energy_anom_ts.data = downwelling_energy_ts.data - control_downwelling_energy_ts.data[month_index]
                big_dict[model]['anom_ts']['downwelling_energy'] = downwelling_energy_anom_ts


if pickle_save:


    def save_big_dict_without_cubes(big_dict, out_path):
        """
        Save big_dict, excluding big_dict[model]['cubes'] for each model.
        """
        print(f'Saving big dicts without cubes to reduce file size...')
        trimmed = deepcopy(big_dict)

        for model in list(trimmed.keys()):
            if isinstance(trimmed[model], dict):
                trimmed[model].pop('cubes', None)
                trimmed[model].pop('ts', None)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, 'wb') as f:
            pickle.dump(trimmed, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    save_big_dict_without_cubes(paleo_big_dict, "saved_dicts/paleo_big_dict_no_cubes.pkl")
    save_big_dict_without_cubes(control_big_dict, "saved_dicts/control_big_dict_no_cubes.pkl")
    save_big_dict_without_cubes(transient_big_dict, "saved_dicts/transient_big_dict_no_cubes.pkl")
    if paleo2:
        save_big_dict_without_cubes(paleo2_big_dict, "saved_dicts/paleo2_big_dict_no_cubes.pkl")
    save_big_dict_without_cubes(crossing_big_dict, "saved_dicts/crossing_big_dict_no_cubes.pkl")
    if paleo2:
        save_big_dict_without_cubes(crossing2_big_dict, "saved_dicts/crossing2_big_dict_no_cubes.pkl")


if pickle_load:
    def load_big_dict(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    print(f'Loading big dicts without cubes from pickles...')
    paleo_big_dict = load_big_dict("saved_dicts/paleo_big_dict_no_cubes.pkl")
    print(f'Loaded {paleo} big dict')
    control_big_dict = load_big_dict("saved_dicts/control_big_dict_no_cubes.pkl")
    print(f'Loaded {control} big dict')
    transient_big_dict = load_big_dict("saved_dicts/transient_big_dict_no_cubes.pkl")
    print(f'Loaded {transient} big dict')
    if paleo2:
        paleo2_big_dict = load_big_dict("saved_dicts/paleo2_big_dict_no_cubes.pkl")
        print(f'Loaded {paleo2} big dict')
    crossing_big_dict = load_big_dict("saved_dicts/crossing_big_dict_no_cubes.pkl")
    print(f'Loaded {paleo} crossing big dict')
    if paleo2:
        crossing2_big_dict = load_big_dict("saved_dicts/crossing2_big_dict_no_cubes.pkl")
        print(f'Loaded {paleo2} crossing big dict')

def add_mmm_to_big_dict(big_dict):
    big_dict['MMM'] = {'clim': {}, 'anom_clim': {}}

    for var_type in ['clim', 'anom_clim']:
        if var_type == 'anom_clim' and big_dict == control_big_dict:
            pass
        else:
            for var in big_dict[models[0]][var_type].keys():
                print(f'Calculating MMM for {var_type} {var}')
                mmm = np.zeros(12)
                for model in models:
                    mmm += np.asarray(big_dict[model][var_type][var].core_data())
                mmm /= len(models)
                big_dict['MMM'][var_type][var] = mmm



print('Adding MMM to absolute climatologies', end ='...')

print(control, end='...')
add_mmm_to_big_dict(control_big_dict)
print(transient, end='...')
add_mmm_to_big_dict(transient_big_dict)
print(paleo, end='...')
add_mmm_to_big_dict(paleo_big_dict)
print(paleo2, end='...')
add_mmm_to_big_dict(paleo2_big_dict)
print(paleo +' Crossing', end='...')
add_mmm_to_big_dict(crossing_big_dict)
print(paleo2 +' Crossing', end='...')
add_mmm_to_big_dict(crossing2_big_dict)
print('Done')
