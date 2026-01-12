import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

# These are the parameters given for the radar data.

c = 299_792_458.0    # Speed of light (m/s)
fs = 18.96e6    # Sample rate (Hz)
PRF = 1679.9    # Pulse repetition frequency (Hz)
lambda_m = 0.0566   # Wavelength (m)

# Use effective velocity for the azimuth compression algorithm.
# Effective velocity adjusts for curved earth geometry.

v_mps = 7138.873751  # Effective velocity
fdc_hz = -300.0   # Doppler centroid in Hz (provided as part of radar parameters)
r_near_m = 821_000.0  # Slant range distance to the first range bin
l_az_m = 10.0  # Antenna length

script_location = os.path.dirname(os.path.abspath(__file__))
USER_FILE = os.path.join(script_location, "ersRC.hw4")

# Number of azimuth lines (given)
N_AZ = 10100
# Number of range bins (given)
N_RNG = 4200
# Assume np.complex64 data type
X = np.fromfile(USER_FILE, dtype=np.complex64)
X = X.reshape(N_AZ, N_RNG)

# The below block of code gives the algorithm for azimuth matched filtering.
# This algorithm was provided by Dr. Ann Chenn, professor at UT Austin.

# Slant range bin spacing
dr = c / (2.0 * fs)

# Slant range distance of each range bin 
range_az = r_near_m + np.arange(N_RNG, dtype=np.float64) * dr

# Doppler centroid in range distance
rdc = np.sqrt(range_az**2 + (fdc_hz * range_az * lambda_m / (2.0 * v_mps))**2)

# Azimuth chirp rate
fR = -2.0 * (v_mps**2) / (lambda_m * rdc)

# Length of time the azimuth chirp will be operating over
tau_az = (rdc * lambda_m) / (v_mps * l_az_m)

# Adjusting the time vector, t', for Doppler shift calculations
n = np.arange(N_AZ, dtype=np.float64)
t = (n - (N_AZ - 1)/2.0) / PRF 

# Create the dynamic reference chirp whose chirp rate changes 
# for each range bin due to Doppler shifts.
# Quadrature component
phase_quad = np.pi * np.outer(t**2, fR)
# In phase component
phase_lin  = (2.0 * np.pi * fdc_hz) * t[:, None]
# Combine the In-phase/Quadrature (I/Q) information
# into a complex exponential, the azimuth reference chirp.
ref_az = np.exp(1j * (phase_quad + phase_lin))

# Create a boolean matrix that checks if 
# the time t is within the domain -tau_az/2 to tau_az/2 .
win = (np.abs(t[:, None]) <= (0.5 * tau_az)[None, :])

# Ignore values in the reference chirp that 
# correspond to time values outside of -tau_az/ 2 to tau_az/2 .
# Make the reference chirp into a matrix of its own in
# order to apply the matched filtering code.
ref_az *= win

# Apply the matched filtering algorithm
X_f = np.fft.fft(X, axis=0)
H_f = np.fft.fft(ref_az, axis=0)
Y_f = X_f * np.conj(H_f)
Y   = np.fft.ifft(Y_f, axis=0)
Y = np.fft.fftshift(Y, axes=0)

# Calculate the multi-looks, where 5 steps are taken 
# in azimuth direction for every one step in the range direction.

looks_azimuth = 5
looks_range = 1
out_rows = N_AZ // looks_azimuth
out_cols = N_RNG // looks_range
image = np.empty((out_rows, out_cols), dtype = np.complex64)

# Use nested for loops to commplete the multilooking process.
# This prevents the outputted image from appearing stretched
# in the azimuth direction.
print("Starting multi-looking process with nested loops...")
start_time = time.time()
for i in range(out_rows):
    r_start = i * looks_azimuth
    r_end = r_start + looks_azimuth
    for j in range(out_cols):
        c_start = j * looks_range
        c_end = c_start + looks_range
        patch = Y[r_start:r_end, c_start:c_end]
        # Compute the average of each patch which
        # becomes a single pixel in the outputted image.
        avg_val = np.mean(patch)
        image[i, j] = avg_val
    # Print every 50 iterations to update user on the loop progress.
    if i % 50 == 0:
        print(f"Processed row {i} of {out_rows}")

end_time = time.time()
print(f"Multi-looking complete! Time taken: {end_time - start_time:.2f} seconds")

# Extract the magnitude information from the complex data.
mag_image = np.abs(image)

# A linear cap of 15_000 yielded the best image quality.
cap = 15000.0
image_capped = np.clip(image, 0.0, cap)
plt.figure()
plt.imshow(
    mag_image,
    aspect='auto',
    origin = 'lower',
    cmap='gray',    
    vmin=0.0, vmax = cap
)
plt.colorbar(label='Amplitude (linear, capped at 15000)')
plt.xlabel('Range bin')
plt.ylabel('Azimuth bin')
plt.title('Multilooked Image (4 looks)')
plt.show()
