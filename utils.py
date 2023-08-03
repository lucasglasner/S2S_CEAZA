#!usr/bin/python3

import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as signal

def haversine(p1,p2):
    """
    Given two points with lat,lon coordinates, compute the distance
    between those points on the surface of the sphere with the haversine formula
    Args:
        p1 (tuple): first point lat,lon
        p2 (tuple): last point lat,lon

    Returns:
        float: distance
    """
    lat1,lon1 = p1
    lat2,lon2 = p2

    lon1,lon2,lat1,lat2 = map(np.deg2rad, [lon1,lon2,lat1,lat2])

    dlon = lon2-lon1
    dlat = lat2-lat1

    a = np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    r = 6371
    return c*r

def distances(data, lon_name='lon', lat_name='lat'):
    """
    This function uses the haversine formula to build a 
    distance vector based on the latitude and longitude
    of the hovmoller data contained in xarray format.
    Args:
        data (xarray.Dataset): 
        Data set with the coastal or coastnorth hovmoller

    Returns:
        np.array: distances in km
    """
    distance = []
    for i in range(len(data[lon_name].values)-1):
        lon1,lat1 = data[lon_name].values[i], data[lat_name].values[i]
        lon2,lat2 = data[lon_name].values[i+1], data[lat_name].values[i+1]
        dist = haversine(p1=(lat2,lon2), p2=(lat1,lon1))
        distance.append(dist)
    distance = np.hstack([np.array([0]),np.array(distance)])
    distance = np.cumsum(distance)
    return distance
    
    
def coastcoords(mask):
    """
    For a given KW pathway mask (S2S, GLORYS, ASCAT, etc)
    return the coordinates of the coastal latitudes
    of the southern and northern hemisphere
    """
    # Find longitude of each coastal american pixel
    lon,lat = np.meshgrid(mask.lon,mask.lat)
    coastnorthcoords   = pd.DataFrame((lon[0,:],np.zeros(len(lon[0,:]))), index=['lon','lat']).T
    lonn = []
    lons = []
    for i in range(len(mask.lat)):
        try:
            x = mask.coastmask_north.where(mask.coastmask_north==1)
            x = x.isel(lat=i).dropna('lon')[-1].lon.item()
            lonn.append(x)
        except:
            lonn.append(np.nan)
            pass
        
        try:
            x = mask.coastmask_south.where(mask.coastmask_south==1)
            x = x.isel(lat=i).dropna('lon')[-1].lon.item()
            lons.append(x)
        except:
            lons.append(np.nan)
            pass
    coastnorthcoords = pd.DataFrame((np.array(lonn),lat[:,0]),index=['lon','lat']).T.dropna()
    coastsouthcoords = pd.DataFrame((np.array(lons),lat[:,0]),index=['lon','lat']).T.dropna()

    coastnorthcoords.index = coastnorthcoords.lat
    coastsouthcoords.index = coastsouthcoords.lat
    return coastnorthcoords, coastsouthcoords

def add_hovmoller_coordinatedata(tropical,coastnorth,coastsouth,mask,variables):
    """
    This function just grabs the tropical, coastnorth and coastsouth hovmollers
    and add some coordinate data (like an index along the path and the distance in km)
    """
    coastnorthcoords, coastsouthcoords = coastcoords(mask)
    # Assign the position along lons as a new spatial dimension/coordinate
    tropical = tropical.assign_coords({'index':('lon',range(len(tropical.lon)))})
    tropical = tropical.swap_dims({'lon':'index'}) # Make the position in the grid as the main coordinate
    tropical = tropical.assign_coords({'lat':('index', np.zeros(len(tropical.index)))}) # Add latitudes

    # Same but for coastnorth array
    coastnorth = coastnorth.assign_coords({'index':('lat', 1+tropical.index[-1].item()+np.arange(len(coastnorth.lat)))}) # Add the position coordinate as the continuity of the coastnorth one
    coastnorth = coastnorth.swap_dims({'lat':'index'})
    coastnorth = coastnorth.assign_coords({'lon':('index', coastnorthcoords.lon.values)})

    # Same but for coastsouth array
    coastsouth = coastsouth.assign_coords({'index':('lat', 1+tropical.index[-1].item()+np.arange(len(coastsouth.lat)))}) # idem
    coastsouth = coastsouth.swap_dims({'lat':'index'})
    coastsouth = coastsouth.assign_coords({'lon':('index', coastsouthcoords.lon.values)})
    
    # Assign distance as new coordinate for the tropical hovmoller
    tropical = tropical.assign_coords({'distance':('index',distances(tropical))})
    tropical = tropical.swap_dims({'index':'distance'})
    tropical = tropical[variables]

    # Assign distance as new coordinate for the coastnorth hovmoller
    gap = haversine((tropical.lat[-1].item(),tropical.lon[-1].item()),(coastnorth.lat[0].item(),coastnorth.lon[0].item())) # Distance between last pixel of coastnorth hovmoller and fisrst of coastnorth hovmoller
    coastnorth = coastnorth.assign_coords({'distance':('index',tropical.distance[-1].item()+gap+distances(coastnorth))})
    coastnorth = coastnorth.swap_dims({'index':'distance'})
    coastnorth = coastnorth[variables]

    # Assign distance as new coordinate for the coastsouth hovmoller
    gap = haversine((tropical.lat[-1].item(),tropical.lon[-1].item()),(coastsouth.lat[0].item(),coastsouth.lon[0].item())) # Distance between last pixel of coastnorth hovmoller and fisrst of coastnorth hovmoller
    coastsouth = coastsouth.assign_coords({'distance':('index',tropical.distance[-1].item()+gap+distances(coastsouth))})
    coastsouth = coastsouth.swap_dims({'index':'distance'})
    coastsouth = coastsouth[variables]
    return tropical,coastnorth,coastsouth



import scipy.signal as signal
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


def compute_climatology(data, dim='time', order=5, cutoff=1/90,
                        kind='mean'):
    """
    This function computes a daily climatology with the following method:
    1) Transform calendar to 'noleap' (same as remove 29 feb)
    2) Compute the daily average
    3) Repeat the daily (noisy) climatology for 30 years
    4) Smooth with a lowpass filter
    5) Compute again the daily average and return
    
    
    Args:
        data (xarray): dataset or datarray for climatology computing
        dim (str, optional): Name of the time dimension. Defaults to 'time'.
        order (int, optional): Butterworth filter order. Defaults to 5.
        cutoff (float, optional): Cutoff frequency. Defaults to 1/80.
        kind (str): 'mean' or 'std'

    Returns:
        _type_: _description_
    """
    clim = data.convert_calendar('noleap', dim=dim)
    if kind=='mean':
        clim = clim.groupby(dim+'.dayofyear').mean()
    elif kind=='std':
        clim = clim.groupby(dim+'.dayofyear').std()
    else:
        raise ValueError('"kind" parameter must be "mean" of "std"')
    clim = xr.concat([clim]*30, 'dayofyear')
    clim.coords[dim] = ('dayofyear',xr.date_range('2000','2030', calendar='noleap')[:-1])
    clim = clim.swap_dims({'dayofyear':dim}).drop('dayofyear')
    # clim = clim.rolling({dim:120}, center=True).mean()
    clim = filter_xarray(clim, dim, order, cutoff)
    clim = clim.groupby(dim+'.dayofyear').mean()
    return clim

