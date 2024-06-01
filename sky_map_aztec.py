from datetime import datetime
from pytz import timezone, utc

import numpy as np
import matplotlib.pyplot as plt
from aztec_code_generator import AztecCode
from tqdm import tqdm

from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
from skyfield.projections import build_stereographic_projection

# Load celestial data
eph = load('de421.bsp')
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

# Place and time of our location
lat, long = 47.400698, 8.512807  # Zurich, Switzerland
when = '2023-06-14 12:00'

# Convert date string into datetime object
dt = datetime.strptime(when, '%Y-%m-%d %H:%M')
local = timezone("Europe/Zurich")
local_dt = local.localize(dt, is_dst=None)
utc_dt = local_dt.astimezone(utc)

# Define observation time and observer
sun = eph['sun']
earth = eph['earth']
ts = load.timescale()
t = ts.from_datetime(utc_dt)
observer = wgs84.latlon(latitude_degrees=lat, longitude_degrees=long).at(t)
position = observer.from_altaz(alt_degrees=90, az_degrees=0)
ra, dec, distance = observer.radec()
center_object = Star(ra=ra, dec=dec)
center = earth.at(t).observe(center_object)
projection = build_stereographic_projection(center)
field_of_view_degrees = 180.0

# Calculate star positions
star_positions = earth.at(t).observe(Star.from_dataframe(stars))
stars['x'], stars['y'] = projection(star_positions)

# Apply limiting magnitude
limiting_magnitude = 5
bright_stars = (stars.magnitude <= limiting_magnitude)
stars = stars[bright_stars]

# Create noise matrix
matrix_size = 1000
#noise_matrix = np.random.randint(0, 2, (matrix_size, matrix_size))
percentage_0 = 85
percentage_1 = 100 - percentage_0
noise_matrix = np.random.choice([0, 1], size=(matrix_size, matrix_size), p=[percentage_0 / 100, percentage_1 / 100])

# Function to create Aztec code matrix without padding
def create_aztec_code_matrix(data):
    aztec_code = AztecCode(data)
    aztec_matrix = np.array(aztec_code.matrix)
    return aztec_matrix

# Generate an Aztec code matrix
aztec_code_matrix = create_aztec_code_matrix("https://example.com")
aztec_size = aztec_code_matrix.shape[0]

# Place Aztec codes at star positions
for i in tqdm(range(len(stars)), desc="Embedding Aztec Codes"):
    x, y = stars['x'].iloc[i], stars['y'].iloc[i]
    if not np.isnan(x) and not np.isnan(y):
        x_idx = int((x + 1) / 2 * (matrix_size - 1))
        y_idx = int((y + 1) / 2 * (matrix_size - 1))
        
        x_start = x_idx - aztec_size // 2
        y_start = y_idx - aztec_size // 2
        x_end = x_start + aztec_size
        y_end = y_start + aztec_size
        
        # Check if the Aztec code will fit within the noise matrix
        if x_start >= 0 and y_start >= 0 and x_end <= matrix_size and y_end <= matrix_size:
            noise_matrix[x_start:x_end, y_start:y_end] = aztec_code_matrix

# Display the result
plt.imshow(noise_matrix, cmap='gray')
plt.axis('off')
plt.show()
