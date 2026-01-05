import numpy as np
import matplotlib.pyplot as plt 
import os
import time

# Range Samples (columns)
width = 6144
# Azimuth Lines (rows)   
lines = 12000  
script_location = os.path.dirname(os.path.abspath(__file__))
filename1 = os.path.join(script_location, "slc1.dat")
filename2 = os.path.join(script_location, "slc2.dat")

# Assume data type is np.complex64. If the code gives good 
# images, then this assumption is correct.
raw_data1 = np.fromfile(filename1, dtype=np.complex64)
slc1 = raw_data1.reshape((lines, width))
raw_data2 = np.fromfile(filename2, dtype = np.complex64)
slc2 = raw_data2.reshape((lines, width))

# Perform cross correlation on the coregistered slc 
# (single look complex) images. This shows how 
# "out of sync" the two images are with 
# very fine precision.
Interf = slc1 * np.conj(slc2)

# Initialize an empty matrix to multi look the interferogram.
# The number of looks to take in the range and azimuth directions
# was given based on ground pixel resolution.
looks_range = 4
looks_azimuth = 16
out_rows = lines // looks_azimuth
out_cols = width // looks_range
Int_final = np.empty((out_rows, out_cols), dtype = np.complex64)

# Sanity check to make sure the interferogram takes a reasonable 
# amount of time to generate.
print("Starting multi-looking process with nested loops...")
start_time = time.time()

# Use a nested loop to perform the multi-looking process.
for i in range(out_rows):
    # r_start (rows start) & r_end (rows end)
    r_start = i * looks_azimuth
    r_end = r_start + looks_azimuth
    for j in range(out_cols):
        # c_start (columns start) & c_end (columns end)
        c_start = j * looks_range
        c_end = c_start + looks_range
        patch = Interf[r_start:r_end, c_start:c_end]
        avg_val = np.mean(patch)
        Int_final[i, j] = avg_val
    # Sanity check, print the row number every 50 iterations to make sure 
    # it is generating correctly.
    if i % 50 == 0:
        print(f"Processed row {i} of {out_rows}")

end_time = time.time()
print(f"Multi-looking complete! Time taken: {end_time - start_time:.2f} seconds.")
# Verify the final shape of the multi-looked matrix.
print(f"Int_final shape: {Int_final.shape}")

# Extract the phase information
phase_img = np.angle(Int_final)
# Display the interferogram
plt.figure(figsize=(12, 8))
plt.imshow(phase_img, cmap='jet', aspect='auto', interpolation='nearest')
plt.colorbar(label='Phase (radians)')
plt.title('Multi-looked Interferogram Phase\n(4 Range Looks x 16 Azimuth Looks)')
plt.xlabel('Range (Samples)')
plt.ylabel('Azimuth (Lines)')
plt.show()

# Open up the DEM (digital elevation map) file.
# We will display this pure float file as a 
# digital elevation map.
dem = os.path.join(script_location, "slc.dem")
dem_file_size = os.path.getsize(dem)

# Each pixel in the DEM map corresponds to the elevation
# of a pixel in the slc files. Thus, it has the same size
# (width * lines) as the slc images.
expected_bytes_float32 = width * lines * 4
expected_bytes_float64 = width * lines * 8

# Confirm whether to process the data as  
# 4 byte float32 data type or 8 byte float64 data type.
if dem_file_size == expected_bytes_float32:
    print("The file has float32 data type")
    dtype_guess = np.float32
elif dem_file_size==expected_bytes_float64:
    print("The file has float64 data type")
    dtype_guess = np.float64

dem_data = np.fromfile(dem, dtype= dtype_guess)
dem_final = dem_data.reshape((lines, width))
# Plot the DEM map
plt.figure(figsize = (10,8))
plt.imshow(dem_final, cmap = "terrain", aspect = "auto")
plt.colorbar(label = "Elevation (meters)")
plt.title("DEM Map of Sample Region")
plt.xlabel("Range")
plt.ylabel("Aziumth")
plt.show()

# This block of code is to calculate the unit vectors from
# the satellite to the elevated point at each pixel
# location and to the same pixel location but at zero elevation.
# I used curved earth geometry for this construction. Below 
# are the parameters for this radar geometry. 
earth_radius = 6_343_837.13
# Satellite height
s_height = 700000
# Speed of light (c)
c = 299_792_458
# Range sample rate (fs)
fs = 32e6
# Range to first pixel (r0)
r0 = 741489
wavelength = 0.236057

# The unit vector changes for each range bin.
# Thus, create a range vector for vectorized calculations.
# This range vector contains slant range distances 
# to all of the range bins.
dr = c / (2*fs)
rng_bin = np.arange(width)
rng_vec = r0 + rng_bin * dr
# Reshape the range vector into a row vector.
rng_vec = rng_vec.reshape(1, -1)
# This is the pure elevation of each pixel.
z = dem_final

# Using trigonometric identities, the equation
# cos_topo_num / cos_topo_den gives the cosine of the look angle.
# Since the z-direction is pointing down, it is assigned 
# a negative value.
# These calculations are for the non-flat earth case.
cos_topo_num = (s_height + earth_radius)**2 + rng_vec**2 - (z + earth_radius)**2
cos_topo_den = (2*rng_vec)*(s_height + earth_radius)
z_topo = -1*(cos_topo_num / cos_topo_den)
# Use trig idenity to get the y-component.
sin_topo = np.sqrt(1-z_topo**2)
y_topo = sin_topo

# These calculations are for the flat earth (zero elevation) case.
cos_flat_num = (s_height + earth_radius)**2 + rng_vec**2 - (0 + earth_radius)**2
cos_flat_den = (2*rng_vec)*(s_height + earth_radius)
z_flat = -1*(cos_flat_num / cos_flat_den)

sin_flat = np.sqrt(1-z_flat**2)
y_flat = sin_flat

# For non flat earth, the unit vector = (y_topo, z_topo)
# For flat earth, the unit vector = (y_flat, z_flat)

# The slc.baseline file provides the physical distance between
# the two orbiting satellites in the y and z direction for each
# azimuth line of data. By performing a dot product, we can 
# isolate the phase difference between eacb satellite's data based
# only on topography.
baseline_path = os.path.join(script_location, "slc.baseline")
baseline_data = np.loadtxt(baseline_path)

# Extract the By (baseline y) and Bz (baseline z) vectors, 
# respectively, and reshape them into column vectors.
By = baseline_data[:, 1].reshape(-1, 1)
Bz = baseline_data[:, 2].reshape(-1, 1)

# Perform a dot product between the baseline vector 
# and both the topographic and flat earth unit vectors.
proj_topo = (By * y_topo) + (Bz * z_topo)
proj_flat = (By * y_flat) + (Bz * z_flat)

# The difference between these is used to find the 
# range difference between the two satellites and each
# pixel.
delta_r_diff = proj_topo - proj_flat

# This formula relates the measured range difference
# to the phase difference.
simulated_topo_phase = - (4 * np.pi / wavelength) * delta_r_diff

# Plot the simulated phase difference map, purely from topography.
plt.figure(figsize=(10, 8))
plt.imshow(simulated_topo_phase, cmap='jet', aspect='auto') 
plt.colorbar(label='Unwrapped Phase (radians)')
plt.title('Simulated Topographic Phase\n(Real Topo - Flat Earth)')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.show()

# This block of code yields the final deformation map.
# By turning the simulated phase difference (simulated_topo_phase)
# into a complex exponential, the phase information can be subtracted
# from the raw interferogram.
sim_phasor_full = np.exp(1j * simulated_topo_phase)

# Subtract the topographic phase information from the raw interferogram. 
# Interf * sim_phasor_full ensures the phase information is subtracted. 
deformation_full_res = Interf * (sim_phasor_full)

# Initialize an empty matrix for the multi-looked deformation map,
# taking 4 looks in the range direction and 16 looks
# in the azimuth direction. 
def_final = np.empty((out_rows, out_cols), dtype = np.complex64)

# Use the same nested loop as was used to muli-look the interferogram. 
# This prevents deformed pixels from appearing in the deformation map.
print("Starting multi-looking process with nested loops...")
start_time = time.time()
for i in range(out_rows):
    r_start = i * looks_azimuth
    r_end = r_start + looks_azimuth
    for j in range(out_cols):
        c_start = j * looks_range
        c_end = c_start + looks_range
        patch = deformation_full_res[r_start:r_end, c_start:c_end]
        avg_val = np.mean(patch)
        def_final[i, j] = avg_val
    if i % 50 == 0:
        print(f"Processed row {i} of {out_rows}")

end_time = time.time()
print(f"Multi-looking complete! Time taken: {end_time - start_time:.2f} seconds.")
# Verify the final shape of the deformation map.
print(f"Int_final shape: {def_final.shape}")

# Extract the phase information
def_phase = np.angle(def_final)
# Display the deformation map
plt.figure(figsize=(12, 8))
plt.imshow(def_phase, cmap='jet', aspect='auto', interpolation='nearest')
plt.colorbar(label='Phase (radians)')
plt.title('Multi-looked Deformation Map\n(4 Range Looks x 16 Azimuth Looks)')
plt.xlabel('Range (Samples)')
plt.ylabel('Azimuth (Lines)')
plt.show()


