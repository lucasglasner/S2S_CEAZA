#!usr/bin/python3

import numpy as np
import pandas as pd
import xarray as xr

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

