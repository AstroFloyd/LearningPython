#!/bin/env python3

import math as m
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.collections import PatchCollection

def rotate(axis, theta):
    """Return a rotation matrix for a rotation over angle theta about axis axis"""
    
    if(axis == 'x' or axis == 'X'):
        rotate = np.array( [ [1,0,0], [0,m.cos(theta),m.sin(theta)], [0,-m.sin(theta),m.cos(theta)] ] )
    elif(axis == 'y' or axis == 'Y'):
        rotate = np.array( [ [m.cos(theta),0,-m.sin(theta)], [0,1,0], [m.sin(theta), 0, m.cos(theta)] ] )
    elif(axis == 'z' or axis == 'Z'):
        rotate = np.array( [ [m.cos(theta),m.sin(theta),0], [-m.sin(theta),m.cos(theta),0], [0,0,1] ] )
    else:
        print("Rotate(): unknown rotation axis: ", axis, " - please specify 'x', 'y' or 'z'")
        rotate = 0
        
    return rotate




corners = [ [0,0], [0,3], [5,3], [5,0] ]
poly = patch.Polygon(corners, alpha=0.5, edgecolor=None)
#plt.gca().add_patch(poly)

circle = patch.Circle([-1,-1], 3, alpha=0.3)
#plt.gca().add_patch(circle)


# Unit circle:
Npoly=4+1
xcirc = np.cos(np.linspace(0., np.pi*2, Npoly))
ycirc = np.sin(np.linspace(0., np.pi*2, Npoly))
zcirc = np.zeros(Npoly)
circle = np.stack( (xcirc,ycirc,zcirc), axis=1 )
print(circle)

theta = np.pi*0.4
phi = np.pi*0.1
circle = np.matmul( circle, rotate('z', theta) )
circle = np.matmul( circle, rotate('x', phi) )

print(circle)
#circle = circle[:,0:2]  # Plot x-y
#circle = circle[:,1:3]  # Plot y-z
circle = circle[:,0:3:2]  # Plot x-z
print(circle)

poly = patch.Polygon(circle*2+[2,3], alpha=0.5, facecolor='b', edgecolor='k')
plt.gca().add_patch(poly)


#plt.xlim(-10,10)
#plt.ylim(-10,10)
plt.axis('equal')
plt.axis('off')

#plt.tight_layout()                      # Use narrow margins
plt.savefig('PatchPerspective.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later
