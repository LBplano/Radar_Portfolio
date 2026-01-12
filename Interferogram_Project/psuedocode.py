# In order to produce the interferogram, I was given the width and length
# of the SAR images as part of the radar information. The below code provides
# an approach to determine both the number of range bins and azimuth lines (in
# pixels) of the SAR images, as well as the data type to interpret each pixel as. 

import numpy as np
import matplotlib.pyplot as plt
import os

script_location = os.path.dirname(os.path.abspath(__file__))
filename1 = os.path.join(script_location, "slc1.dat")

# Find the number of bytes in the file
size_bytes = os.path.getsize(filename1)
print(f"The file size is {size_bytes} bytes")

# The file size is 589_824_000 bytes.
# First, interpret it as np.complex64, so 8 bytes per pixel.
factor = int(size_bytes / 8)

# The below loop finds the factors of 589_824_000 / 8 .
# Print them to populate the terminal window with all of 
# the possible factors.

for i in range(1,factor):
    if factor % i == 0:
        width = int(factor/i)
        print(f"The width is {width} and the length is {i}")

# Probe different factors until a coherent image is formed. 
# For each pair of factors, change the width and lines variables,
# then use imshow to display the image. 
# A trial of guess and check will inform which range of factors are
# reasonable and which are not. 
raw_data = np.fromfile(filename1, dtype = np.complex64)

# This guess and check method yielded 6144 columns and 12000 rows
# as the correct pair of factors to reshape the data into.
width = 6144
lines = 12000

# Extract the magnitude from the complex data, then apply a linear
# cap. This yields better image quality then applying a logarithmic scale.
cap = 600
slc1 = raw_data.reshape((lines, width))
slc1_0 = np.abs(slc1)
slc1_capped = np.clip(slc1_0, 0.0, cap)

plt.figure()
plt.imshow(
    slc1_capped, 
    aspect = 'auto', 
    origin = 'lower', 
    cmap = 'gray', 
    vmin = 0.0, 
    vmax = cap)
plt.colorbar(label = 'Amplitude (linear, capped at )')
plt.show()

# The image displayed is a coherent SAR image. Thus, the assumption
# that the data is complex64 was correct. If the displayed image
# had a consistently "static" appearance, then it would be reasonable
# to attempt this same process but using complex128 as the data type. 