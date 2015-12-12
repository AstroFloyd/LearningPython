#!/usr/bin/env python3

## @file numpy_array_print_formatted.py
#  Formatted (readable, aligned) text print output of a numpy array
#

import numpy as np

## Data array
dat = np.random.random((11,2))*100

# Print the array itself directly:
print()
print(dat)


# Print the array contents as text:
print()
for i in range(np.shape(dat)[0]):
    print("%4.1f %6.2f" % tuple(dat[i,:]))


# For a wide array, you can repeat format groups:
dat = np.random.random((10,11))*100  # Array of random values between 0 and 100
print()
print(dat)  # Lines get truncated on the screen and aredifficult to read
print()
for i in range(10):
    print((4*"%6.2f"+7*"%9.4f") % tuple(dat[i,:]))


    
print()

