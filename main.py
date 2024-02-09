# https://docs.poliastro.space/en/stable/examples/Generating%20orbit%20groundtracks.html

import datetime
import numpy as np
from matplotlib import pyplot as plt
import logging

# Useful for defining quantities
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import (
    GCRS,
    ITRS,
    SphericalRepresentation,
    CartesianRepresentation,
    CartesianDifferential
)

from poliastro.earth import EarthSatellite
from poliastro.twobody import Orbit
from poliastro.util import time_range
from poliastro.core.spheroid_location import cartesian_to_ellipsoidal
from poliastro.bodies import Earth
from poliastro.twobody.sampling import EpochsArray

logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def def_iss_orb():
    from poliastro.examples import iss
    # Build spacecraft instance
    iss_spacecraft = EarthSatellite(iss, None)
    # t_span = time_range(
    #     iss.epoch - 1.5 * u.h, periods=150, end=iss.epoch + 1.5 * u.h
    # )

    return iss_spacecraft.orbit

def def_xleo_orb():

    return Orbit.circular(Earth, 525*u.km, 55*u.deg)


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


# def haversine_grid(lat, lon, lats, lons):
#
#     # Difference in coordinates
#     dlat = np.radians(lats - lat)
#     dlon = np.radians(lons - lon)
#
#     # Haversine formula
#     a = np.sin(dlat / 2) ** 2 + (np.cos(lat) * np.cos(lats) * np.sin(dlon / 2) ** 2)
#     c = np.arcsin(np.sqrt(a))
#
#     # Total distance in kilometers
#     distance = 2 * (Earth.R.value / 1000) * c
#
#     return distance.T


def spherical_cosines(lat, lon, lats, lons):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat, lon, lats, lons])

    distance = np.arccos(np.sin(lat1) * np.sin(lat2) +
                         np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)) * (Earth.R.value / 1000)  # Radius of the Earth in kilometers

    return distance

def _get_raw_coords(orb, t_deltas):
    """Generates raw orbit coordinates for given epochs.

    Parameters
    ----------
    orb : ~poliastro.twobody.Orbit
        Orbit to be propagated
    t_deltas : ~astropy.time.DeltaTime
        Desired observation time

    Returns
    -------
    raw_xyz : numpy.ndarray
        A collection of raw cartesian position vectors
    raw_epochs : numpy.ndarray
        Associated epoch with previously raw coordinates
    """
    # Solve for raw coordinates and epochs
    ephem = orb.to_ephem(EpochsArray(orb.epoch + t_deltas))
    rr, vv = ephem.rv()
    raw_xyz = CartesianRepresentation(
        rr,
        xyz_axis=-1,
        differentials=CartesianDifferential(vv, xyz_axis=-1),
    )
    raw_epochs = ephem.epochs

    return raw_xyz, raw_epochs


def _from_raw_to_ITRS(raw_xyz, raw_obstime):
    """Converts raw coordinates to ITRS ones.

    Parameters
    ----------
    raw_xyz : numpy.ndarray
        A collection of rwa position coordinates
    raw_obstime : numpy.ndarray
        Associated observation time

    Returns
    -------
    itrs_xyz: ~astropy.coordinates.ITRS
        A collection of coordinates in ITRS frame

    """
    # Build GCRS and ITRS coordinates
    gcrs_xyz = GCRS(
        raw_xyz,
        obstime=raw_obstime,
        representation_type=CartesianRepresentation,
    )
    itrs_xyz = gcrs_xyz.transform_to(ITRS(obstime=raw_obstime))

    return itrs_xyz


def propagate_orbit(orb, start_t, end_t):

    #gt = GroundtrackPlotter()
    t_span = time_range(
        Time(start_t.strftime("%Y-%m-%dT%H:%M:%S"), format="isot", scale="utc"),
             periods=interval,
             end=Time(end_t.strftime("%Y-%m-%dT%H:%M:%S"), format="isot", scale="utc")
    )

    t_deltas = t_span - orb.epoch

    raw_xyz, raw_obstime = _get_raw_coords(orb, t_deltas)
    itrs_xyz = _from_raw_to_ITRS(raw_xyz, raw_obstime)
    itrs_latlon = itrs_xyz.represent_as(SphericalRepresentation)

    lat = itrs_latlon.lat.to(u.deg),
    lon = itrs_latlon.lon.to(u.deg)

    return lat, lon


def is_sat_over_target(sat_coords, global_coords, max_off_zenith):

    rad = sat_coords[2] * np.sin(np.radians(max_off_zenith))

    #dist = haversine_grid(global_coords[0], global_coords[1], sat_coords[0], sat_coords[1])
    dist = spherical_cosines(global_coords[0], global_coords[1], sat_coords[0], sat_coords[1])

    return dist <= rad


def gen_coord_grid(north_max, south_max, dlat, dlon):

    # generate earth coords
    lat_vec = np.arange(south_max, north_max, dlat)
    lon_vec = np.arange(-180, 180, dlon)
    lat_grid, lon_grid = np.meshgrid(lat_vec, lon_vec)

    return lat_grid, lon_grid


def calc_MTTA(orbits, time_steps, grid, max_off_zenith):

    logging.debug("Starting run")

    # active_points = np.ones_like(grid[0]) * True
    MTTA = np.zeros_like(grid[0])
    number_access = np.zeros_like(grid[0])
    # new_access = np.ones_like(grid[0]) * False
    last_access = np.ones_like(grid[0]) * False

    lat, lon = propagate_orbit(orbits, time_steps[0], time_steps[-1])

    in_range_grid_n0 = is_sat_over_target((lat[0][0].value, lon[0].value, altitude),
                                       (lat_grid, lon_grid),
                                       max_off_zenith)

    for n, t in enumerate(time_steps):

        if n % 500 == 0:
            logging.info(f"{n} / {len(time_steps)} step complete")

        in_range_grid_n1 = is_sat_over_target((lat[0][n].value, lon[n].value, altitude),
                                           (lat_grid, lon_grid),
                                           max_off_zenith)

        new_access_points = (in_range_grid_n0 != in_range_grid_n1) & in_range_grid_n1
        number_access[new_access_points] += 1

        if np.all(number_access[new_access_points] <= 1):
            last_access_points = (in_range_grid_n0 != in_range_grid_n1) & in_range_grid_n0
            last_access[last_access_points] = t
            continue

        valid_access = new_access_points & last_access

        MTTA[valid_access] += (t - last_access[valid_access]) / (number_access[valid_access] + 1)

        last_access_points = (in_range_grid_n0 != in_range_grid_n1) & in_range_grid_n0
        last_access[last_access_points] = t

        in_range_grid_n0 = in_range_grid_n1

    logger.info("MTTA run complete")

    return MTTA


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    lat_grid, lon_grid = gen_coord_grid(60, -60, 1, 1)
    altitude = 525
    max_off_zenith = 30

    # generate time
    start = "2025-01-01T00:00:00"
    end = "2025-01-02T00:00:00"
    interval = 24*60
    time_list = lintime(start, end, interval) # DOESN'T WORK

    # define orbit
    orb = def_xleo_orb()

    MTTA = calc_MTTA(orb, time_list, (lat_grid, lon_grid), max_off_zenith)

    # plt.imshow(in_range_grid)

    # plt.scatter(lon, lat)
    # plt.xticks(labels=lon_vec,)
    # plt.show()




