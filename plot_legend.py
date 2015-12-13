#!/usr/bin/env python

## @file plot_legend.py
#  Python example for using a legend in a plot using matplotlib
#


import matplotlib.pyplot as plt
import numpy as np

## x array
x = np.arange(0.0, 2.0, 0.01)               # Array 0-2 with steps of 0.01

## Sine array
ys = np.sin(2*np.pi*x)                      # sin(2pi x)
## Cosine array
yc = np.cos(2*np.pi*x)                      # cos(2pi x)

#plt.xlim(0,2.7)                             # Extend the horizontal plot range (before plotting) to create room for the legend box
plt.ylim(-1.05,1.4)                             # Extend the horizontal plot range (before plotting) to create room for the legend box

line_sin, = plt.plot(x, ys, label="Sine")   # Plot sine
line_cos, = plt.plot(x, yc, label="Cosine") # Plot cosine
#plt.plot(x, ys, 'b-o',   x, yc, 'r-s') # Plot both

plt.legend(handles=[line_sin,line_cos])     # Create legend

plt.xlabel('x axis')                        # Label the horizontal axis
plt.ylabel('y axis')                        # Label the vertical axis
plt.title('A sine and a cosine')            # Plot title
plt.grid(True)                              # Plot a grid

plt.tight_layout()                          # Use narrow margins
plt.savefig("plot_legend.png")              # Save the plot as png
#plt.show()                                  # Show the plot to screen


