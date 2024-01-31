# https://docs.poliastro.space/en/stable/examples/Generating%20orbit%20groundtracks.html

# Useful for defining quantities
from astropy import units as u

# Earth focused modules, ISS example orbit and time span generator
from poliastro.earth import EarthSatellite
from poliastro.earth.plotting import GroundtrackPlotter
from poliastro.examples import iss
from poliastro.util import time_range

def main():
    # Build spacecraft instance
    iss_spacecraft = EarthSatellite(iss, None)
    t_span = time_range(
        iss.epoch - 1.5 * u.h, periods=150, end=iss.epoch + 1.5 * u.h
    )

    # Generate an instance of the plotter, add title and show latlon grid
    gp = GroundtrackPlotter()
    gp.update_layout(title="International Space Station groundtrack")

    # Plot previously defined EarthSatellite object
    gp.plot(
        iss_spacecraft,
        t_span,
        label="ISS",
        color="red",
        marker={
            "size": 10,
            "symbol": "triangle-right",
            "line": {"width": 1, "color": "black"},
        },
    )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main()


