#!/usr/bin/env python3

## @file numpy_array_print_formatted.py
#  Formatted (readable, aligned) text print output of a numpy array
#

import numpy as np
import matplotlib.pyplot as plt

## Data array
dat = np.loadtxt("plot_data.dat")  # Reads the 11x2 array

# Print the array directly itself:
print()
print(dat)


# Print the array contents as text:
print()
for i in range(11):
    print("%4.1f %6.2f" % tuple(dat[i,:]))


# For a wide array, you can repeat format groups:
print()
for i in range(2):
    print(("%6.2f"*4 + "%7.2f"*6 + "%8.2f") % tuple(dat[:,i]))

print()

