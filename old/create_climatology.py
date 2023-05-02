from glob import glob
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.signal as signal
from scipy.interpolate import interp1d
import os


def filter_timeseries(ts, order, cutoff, btype='lowpass', fs=1, **kwargs):
    """Given an array, this function apply a butterworth (high/low pass) 
    filter of the given order and cutoff frequency.
    For example:
    If 'ts' is a timeseries of daily samples, filter_timeseries(ts,3,1/20)
    will return the series without the 20 days or less variability using an
    order 3 butterworth filter. 
    In the same way, filter_timeseries(ts,3,1/20, btype='highpass') will
    return the series with only the 20 days or less variability.

    Args:
        ts (array_like): timeseries or 1D array to filter
        order (int): _description_
        cutoff (array_like): Single float for lowpass or highpass filters, 
        arraylike for bandpass filters.
        btype (str, optional): The type of filter. Defaults to 'lowpass'.
        fs (int): Sampling frequency. Defaults to 1
        **kwargs are passed to scipy.signal.filtfilt

    Returns:
        output (array): Filtered array
    """
    mask = np.isnan(ts)
    nans = np.ones(len(ts))*np.nan
    if mask.sum()==len(ts):
        return nans
    else:
        b, a = signal.butter(order, cutoff, btype=btype, fs=fs)
        filt=signal.filtfilt(b, a, ts[~mask], **kwargs)
        output=np.ones(len(ts))*np.nan
        output[np.where(~mask)] = filt
        return output
    
def filter_xarray(data, dim, order, cutoff, btype='lowpass', parallel=False, fs=1):
    """Given a 3d DataArray, with time and spatial coordinates, this function apply
    the 1D function filter_timeseries along the time dimension, filter the complete
    xarray data.

    Args:
        data (XDataArray): data
        dim (str): name of the time dimension
        order (int): butterworth filter order
        cutoff (array_like): if float, the cutoff frequency, if array must be the
                            [min,max] frequencys for the bandpass filter.
        btype (str, optional): {lowpass,highpass,bandpass}. Defaults to 'lowpass'.
        parallel (bool, optional): If parallelize with dask. Defaults to False.
        fs (int, optional): Sampling frequency. Defaults to 1.

    Returns:
        XDataArray: filtered data
    """
    if parallel:
        dask='parallelized'
    else:
        dask='forbidden'
    filt = xr.apply_ufunc(filter_timeseries, data, order, cutoff, btype, fs,
                          input_core_dims=[[dim],[],[],[],[]],
                          output_core_dims=[[dim]],
                          exclude_dims=set((dim,)),
                          keep_attrs=True,
                          vectorize=True, dask=dask)
    filt[dim] = data[dim]
    return filt



def climatology1d(data, times, period=365, nharmonics=3):
    """
    This function computes the annual cycle of a time series 
    taking advantage of the fourier series decomposition and 
    that seasonal changes are equal to the 365 time scale 
    variability. 
    
    By defaults the function assumes the input data has a 
    daily sampling, if not modify "period" argument properly. 

    Args:
        data (array): (n, ) dimensional array with time-ordered data
        times (array):(n, ) dimensional array of time objects
        period (int, optional): Period to remove. Defaults to 365.
        nharmonics (int, optional): Number of harmonics related to 
        the fundamental frequency to remove. Defaults to 3.

    Returns:
        array: (365, ) dimensional array with the climatological
        value of each day.
        
    Reference: Statistical Methods in the Atmospheric Sciences
              Daniel S. Wilks. Part II Univariate Statistics, 
              Chapter 9.4: Frequency Domain - Harmonic Analysis
              
    Example 9.8: Transforming a Cosine Wave to Represent an Annual Cycle
    Section 9.4.3: Estimation of the Amplitude and Phase of a Single Harmonic
    Section 9.4.4: Higher Harmonics
    Example 9.11: A more Complicated Annual Cycle
    """
    
    # If data has nans return full nan vector
    if np.isnan(data).sum()!=0:
        return np.nan*np.ones(365)
    # Transform data to pandas timeseries and find sample frequency
    data   = pd.Series(data, index = pd.to_datetime(times))
    # Remove leap day of leap years and NaNs
    data   = data[~((data.index.month==2)&(data.index.day==29))]
    # Number of timestemps
    n      = len(data) 
    # Compute sample mean
    clim  = np.mean(data)*np.ones(n)
    for k in range(nharmonics):
        # Target frequency
        freq       = 2*np.pi*(k+1)*np.arange(1,n+1)/period
        # Compute harmonic coefficient as a "naive" discrete
        # fourier transform
        Ak        = (2/n)*np.sum(data*np.cos(freq)) 
        Bk        = (2/n)*np.sum(data*np.sin(freq)) 
        # Add up harmonics to the annual cycle
        clim      = clim+Ak*np.cos(freq)+Bk*np.sin(freq)
    clim = pd.Series(clim, index=data.index).to_xarray().rename({'index':'time'})
    clim = clim.to_dataset(name='_').convert_calendar('noleap')['_']
    clim = clim.groupby("time.dayofyear").mean()
    return clim.to_series()

def climatology_xarray(data, dim, period=365, nharmonics=3, parallel=False, **kwargs):
    """
    Given a 3d DataArray, with time and spatial coordinates, this function apply
    the 1D function climatology1d along the time dimension return the spatial
    climatology.

    Args:
        data (_type_): _description_
        dim (_type_): _description_
        period (int, optional): _description_. Defaults to 365.
        nharmonics (int, optional): _description_. Defaults to 3.
        parallel (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if parallel:
        dask='parallelized'
    else:
        dask='forbidden'
    clim = xr.apply_ufunc(climatology1d, data, kwargs={'times':data.time,
                                                       'period':period,
                                                       'nharmonics':nharmonics},
                          input_core_dims=[[dim],],
                          output_core_dims=[[dim]],
                          exclude_dims=set((dim,)),
                          keep_attrs=True,
                          vectorize=True, dask=dask,
                          dask_gufunc_kwargs={'allow_rechunk':True},
                          **kwargs)
    clim = clim.rename({'time':'dayofyear'})
    clim.coords['dayofyear'] = np.arange(1, len(clim.dayofyear)+1)
    return clim



print('loading glorys')
glorys12      = xr.open_mfdataset('data/GLORYS12V1/SST/*').load()
glorys12      = glorys12.sel(time=~((glorys12.time.dt.month==2)&(glorys12.time.dt.day==29)))

print('computing climatology')
glorys12_clim = climatology_xarray(glorys12, 'time')

print('saving to netcdf')
glorys12_clim.to_netcdf('data/GLORYS12V1/SST_CLIMATOLOGY.nc')
