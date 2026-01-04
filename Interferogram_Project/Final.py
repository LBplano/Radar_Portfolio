import numpy as np
import matplotlib.pyplot as plt 
import os
import time

#Part (a)
width = 6144    #Range samples
lines = 12000   #Azimuth lines
script_location = os.path.dirname(os.path.abspath(__file__))
filename1 = os.path.join(script_location, "slc1.dat")
filename2 = os.path.join(script_location, "slc2.dat")
#Load and reshape slc1.dat
raw_data1 = np.fromfile(filename1, dtype=np.complex64)
slc1 = raw_data1.reshape((lines, width))
#Load and reshape slc2.dat
raw_data2 = np.fromfile(filename2, dtype = np.complex64)
slc2 = raw_data2.reshape((lines, width))
#Perfrom cross correlation on the coregistered slc images
Interf = slc1 * np.conj(slc2)
#Initialize an empty matrix to multi look the interferogram
looks_range = 4
looks_azimuth = 16
out_rows = lines // looks_azimuth
out_cols = width // looks_range
Int_final = np.empty((out_rows, out_cols), dtype = np.complex64)
#Sanity check to make sure the interferogram takes a reasonable 
#amount of time to generate
print("Starting multi-looking process with nested loops...")
start_time = time.time()
#Use nested liip structure to perform the multi-looking
for i in range(out_rows):
    r_start = i * looks_azimuth
    r_end = r_start + looks_azimuth
    for j in range(out_cols):
        c_start = j * looks_range
        c_end = c_start + looks_range
        patch = Interf[r_start:r_end, c_start:c_end]
        avg_val = np.mean(patch)
        Int_final[i, j] = avg_val
    #Sanity check, print row every 50 iterations to make sure 
    #it's generating correctly
    if i % 50 == 0:
        print(f"Processed row {i} of {out_rows}")
end_time = time.time()
print(f"Multi-looking complete! Time taken: {end_time - start_time:.2f} seconds.")
#Verify the final shape is (750, 1536)
print(f"Int_final shape: {Int_final.shape}")
#Extract the phase information
phase_img = np.angle(Int_final)
#Display the Interferogram
plt.figure(figsize=(12, 8))
plt.imshow(phase_img, cmap='jet', aspect='auto', interpolation='nearest')
plt.colorbar(label='Phase (radians)')
plt.title('Multi-looked Interferogram Phase\n(4 Range Looks x 16 Azimuth Looks)')
plt.xlabel('Range (Samples)')
plt.ylabel('Azimuth (Lines)')
plt.show()

#Part (b)
dem = os.path.join(script_location, "slc.dem")
#Confirm whether to process the data as the 
#4 byte float32 data type or the 8 byte float64 data type
dem_file_size = os.path.getsize(dem)
expected_bytes_float32 = width * lines * 4
expected_bytes_float64 = width * lines * 8

if dem_file_size == expected_bytes_float32:
    print("The file has float32 data type")
elif dem_file_size==expected_bytes_float64:
    print("The file has float64 data type")

#Confirmed that it is float32
dem_data = np.fromfile(dem, dtype=np.float32)
dem_final = dem_data.reshape((lines, width))
#Plot the DEM map
plt.figure(figsize = (10,8))
plt.imshow(dem_final, cmap = "terrain", aspect = "auto")
plt.colorbar(label = "Elevation (meters)")
plt.title("DEM Map of Hawaii Topography")
plt.xlabel("Range")
plt.ylabel("Aziumth")
plt.show()

#Part (c)
#List the parameters
re = 6_343_837.13
h = 700000
c = 299_792_458
fs = 32e6
r0 = 741489
wavelength = 0.236057
#Create range vector for vectorized calculations
#Contains slant range distances to all of the range bins
dr = c / (2*fs)
rng_bin = np.arange(width)
rng_vec = r0 + rng_bin * dr
#Reshape into a row vector
rng_vec = rng_vec.reshape(1, -1)
z = dem_final

#For non-flat earth
cos_topo_num = (h + re)**2 + rng_vec**2 - (z+re)**2
cos_topo_den = (2*rng_vec)*(h+re)
z_topo = -1*(cos_topo_num / cos_topo_den)

sin_topo = np.sqrt(1-z_topo**2)
y_topo = sin_topo

#For flat earth case
cos_flat_num = (h + re)**2 + rng_vec**2 - (0+re)**2
cos_flat_den = (2*rng_vec)*(h+re)
z_flat = -1*(cos_flat_num / cos_flat_den)

sin_flat = np.sqrt(1-z_flat**2)
y_flat = sin_flat
#Non flat earth, unit vector = (y_topo, z_topo)
#Flat earth, unit vector = (y_flat, z_flat)

#Part (d)
baseline_path = os.path.join(script_location, "slc.baseline")
baseline_data = np.loadtxt(baseline_path)
#Extract the By and Bz vectors, respectively, and reshape them
#into column vectors
By = baseline_data[:, 1].reshape(-1, 1)
Bz = baseline_data[:, 2].reshape(-1, 1)
#Perform dot-product for both the topo and flat earth
#unit vectors
proj_topo = (By * y_topo) + (Bz * z_topo)
proj_flat = (By * y_flat) + (Bz * z_flat)
delta_r_diff = proj_topo - proj_flat
#Formula for relating phase difference to the measured range difference
simulated_topo_phase = - (4 * np.pi / wavelength) * delta_r_diff
#Plot the simulated phase difference map, purely topography
plt.figure(figsize=(10, 8))
plt.imshow(simulated_topo_phase, cmap='jet', aspect='auto') 
plt.colorbar(label='Unwrapped Phase (radians)')
plt.title('Simulated Topographic Phase\n(Real Topo - Flat Earth)')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.show()

#Part (e)
#Turn the simulated phase difference into a complex exponential
#so that we can subtract the phase information from the raw interferogram, which
#also contains data as a complex exponential
sim_phasor_full = np.exp(1j * simulated_topo_phase)
#Initially, I had Interf * np.conj(sim_phasor_full), but only after
#I switched to complex multiplication, that is, adding the phase 
#information, did I get the deformation map
deformation_full_res = Interf * (sim_phasor_full)
def_final = np.empty((out_rows, out_cols), dtype = np.complex64)
#Use same nested loop code to perform the multi looking so that 
#we don't have deformed pixels in the deformation map
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
print(f"Int_final shape: {def_final.shape}")
print(f"Sample value at [0,0]: {def_final[0,0]}")
#Extract the phase information
def_phase = np.angle(def_final)
#Display the deformation map
plt.figure(figsize=(12, 8))
plt.imshow(def_phase, cmap='jet', aspect='auto', interpolation='nearest')
plt.colorbar(label='Phase (radians)')
plt.title('Multi-looked Deformation Map\n(4 Range Looks x 16 Azimuth Looks)')
plt.xlabel('Range (Samples)')
plt.ylabel('Azimuth (Lines)')
plt.show()


