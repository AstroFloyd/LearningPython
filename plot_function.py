#!/usr/bin/env python

## @file plot_function.py
#  Python example for plotting a simple function using matplotlib
#


import matplotlib.pyplot as plt
import numpy as np

## Time array
t = np.arange(0.0, 2.0, 0.01)     # Array 0-2 with steps of 0.01

## Amplitude array
s = np.sin(2*np.pi*t)             # sin(2pi t)
plt.plot(t, s)                    # Plot

plt.xlabel('time (s)')            # Label the horizontal axis
plt.ylabel('amplitude')           # Label the vertical axis
plt.title('Plot title')           # Plot title
plt.grid(True)                    # Plot a grid
plt.savefig("plot_function.png")  # Save the plot as png
plt.show()                        # Show the plot to screen

