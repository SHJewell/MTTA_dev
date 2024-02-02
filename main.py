# https://docs.poliastro.space/en/stable/examples/Generating%20orbit%20groundtracks.html

import datetime
import numpy as np
from matplotlib import pyplot as plt

# Useful for defining quantities
from astropy import units as u
from astropy.time import Time

# Earth focused modules, ISS example orbit and time span generator
from poliastro.earth import EarthSatellite
from poliastro.earth.plotting import GroundtrackPlotter
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.examples import iss
from poliastro.util import time_range
from poliastro.core.spheroid_location import cartesian_to_ellipsoidal
from poliastro.bodies import Earth
from poliastro.twobody.propagation import propagate

def def_orb():
    # Build spacecraft instance
    iss_spacecraft = EarthSatellite(iss, None)
    # t_span = time_range(
    #     iss.epoch - 1.5 * u.h, periods=150, end=iss.epoch + 1.5 * u.h
    # )

    return iss_spacecraft.orbit

def prop_and_calc(orb, dt):

    atime = Time(dt)

    propd_orbit = orb.propagate(atime - orb.epoch)

    r, v = propd_orbit.rv()

    return cartesian_to_ellipsoidal(Earth.R, Earth.R_polar, *r.to(u.m).value)
    #return cartesian_to_ellipsoidal(r[0].to(u.m).value, r[1].to(u.m).value, r[2].to(u.m).value)


def lintime(start_time, end_time, interval_minutes):
    # Convert start and end times to datetime objects if they are not already
    if isinstance(start_time, str):
        start_time = datetime.datetime.fromisoformat(start_time)
    if isinstance(end_time, str):
        end_time = datetime.datetime.fromisoformat(end_time)

    # Calculate the number of intervals
    delta = end_time - start_time
    total_intervals = int(delta.total_seconds() / (interval_minutes * 60))

    # Generate the list of times
    times = [start_time + datetime.timedelta(minutes=interval_minutes) * i for i in range(total_intervals + 1)]

    return times


def haversine_grid(lat, lon, lats, lons):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Difference in coordinates
    dlat = lats - lat
    dlon = lons - lon

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Total distance in kilometers
    distance = R * c

    return distance.T


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    lat_list, lon_list = [], []

    lat_vec = np.radians(np.arange(-60, 60, 1))
    lon_vec = np.radians(np.arange(-180, 180, 1))
    lat_grid, lon_grid = np.meshgrid(lat_vec, lon_vec)

    start = "2013-03-18T10:30:00"
    end = "2024-03-18T13:30:00"
    interval = 150
    time_list = lintime(start, end, interval)

    orb = def_orb()

    for t in time_list:

        lat, lon, _ = prop_and_calc(orb, t)
        lat_list.append(np.degrees(lat))
        lon_list.append(np.degrees(lon))

        dist = haversine_grid(lat, lon, lat_grid, lon_grid)

        # plt.imshow(dist, extent=(-180, 180, 60, -60))
        # plt.scatter(np.degrees(lon), np.degrees(lat))
        # plt.show()


    plt.scatter(lon_list, lat_list)
    plt.xticks(labels=lon_vec,)
    plt.show()


