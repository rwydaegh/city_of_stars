from datetime import datetime
from pytz import timezone, utc
from aztec_code_generator import AztecCode

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_interactions import panhandler, zoom_factory

from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
from skyfield.projections import build_stereographic_projection

# load celestial data

# de421 shows position of earth and sun in space
eph = load('de421.bsp')

# hipparcos dataset contains star location data
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

# place and time of our location 
lat, long = 47.400698, 8.512807 # Zurich, Switzerland
when = '2023-06-14 12:00' # When we said 'I love you' to each other

# convert date string into datetime object
dt = datetime.strptime(when, '%Y-%m-%d %H:%M')

# define datetime and convert to utc based on our timezone
local = timezone("Europe/Zurich")

# get UTC from local timezone and datetime
local_dt = local.localize(dt, is_dst=None)
utc_dt = local_dt.astimezone(utc)

# find location of earth and sun and set the observer position
sun = eph['sun']
earth = eph['earth']

# define observation time from our UTC datetime
ts = load.timescale()
t = ts.from_datetime(utc_dt)

# define an observer using the world geodetic system data
observer = wgs84.latlon(latitude_degrees=lat, longitude_degrees=long).at(t)

# define the position in the sky where we will be looking
position = observer.from_altaz(alt_degrees=90, az_degrees=0)

# center the observation point in the middle of the sky
ra, dec, distance = observer.radec()
center_object = Star(ra=ra, dec=dec)

# find where our center object is relative to earth and build a projection with 180 degree view
center = earth.at(t).observe(center_object)
projection = build_stereographic_projection(center)
field_of_view_degrees = 180.0

# calculate star positions and project them onto a plain space
star_positions = earth.at(t).observe(Star.from_dataframe(stars))
stars['x'], stars['y'] = projection(star_positions)


chart_size = 10
max_star_size = 100
limiting_magnitude = 10

bright_stars = (stars.magnitude <= limiting_magnitude)
magnitude = stars['magnitude'][bright_stars]

fig, ax = plt.subplots(figsize=(chart_size, chart_size))
    
border = plt.Circle((0, 0), 1, color='navy', fill=True)
ax.add_patch(border)

marker_size = max_star_size * 10 ** (magnitude / -2.5)

ax.scatter(stars['x'][bright_stars], stars['y'][bright_stars],
           s=marker_size, color='white', marker='.', linewidths=0, 
           zorder=2)

# now add the sun
'''
sun_position = earth.at(t).observe(sun).apparent()
sun_x, sun_y = projection(sun_position)
sun_marker_size = 100 * 10 ** (magnitude.min() / 2 / -2.5)
sun_marker = plt.Circle((sun_x, sun_y), sun_marker_size, color='yellow', zorder=3)
plt.gca().add_patch(sun_marker)
'''

horizon = Circle((0, 0), radius=1, transform=ax.transData)
for col in ax.collections:
    col.set_clip_path(horizon)


# other settings
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.axis('off')

# Make the plot interactive
zoom_factory(ax)
panhandler(fig)

plt.show()