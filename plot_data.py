#!/usr/bin/env python3

## @file plot_data.py
#  Python example for reading data from a text file and plotting them using matplotlib
#

import numpy as np
import matplotlib.pyplot as plt

## Data array
dat = np.loadtxt("plot_data.dat")  # Reads the 11x2 array

# plt.plot(dat)                    # Would plot two lines, each vs. the array index
#plt.plot(dat[:,0],dat[:,1])        # Plot column 2 as a function of column 1

plt.errorbar(dat[:,0], dat[:,1], fmt='rs-',
    xerr=(dat[:,0]+1)**(-0.1), yerr=dat[:,0]+1)  # Plot error bars, a red line and squares
plt.plot(dat[:,0],dat[:,1], 'bs')                # Use blue iso red squares

plt.xlabel('time (s)')             # Label the horizontal axis
plt.ylabel('time squared')         # Label the vertical axis
plt.title('Plot title')            # Plot title
plt.grid(True)                     # Plot a grid

plt.tight_layout()                 # Use narrow margins
plt.savefig("plot_data.png")       # Save the plot as png
#plt.show()                         # Show the plot to screen
