#!/bin/env python

# https://stackoverflow.com/a/32427177/1386750

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from astroconst import r2d,d2r

# Choose projection:
vpAlt = 10.0 * d2r
vpAz  = 80.0 * d2r


# Setup plot:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# ### SPHERE ###
# Create a sphere:
r = 1
phi   = np.linspace(0, 2*np.pi, 100)  # Azimuthal coordinate
theta = np.linspace(0,   np.pi, 100)  # Altitude coordinate

x = r * np.outer(np.cos(phi), np.sin(theta))
y = r * np.outer(np.sin(phi), np.sin(theta))
z = r * np.outer(np.ones(np.size(phi)), np.cos(theta))

# Plot sphere surface:
ax.plot_surface(x, y, z,  rstride=2, cstride=4, color='b', linewidth=0, alpha=0.5)


# ### EQUATOR ###
# Plot whole equator, dashed:
ax.plot(np.sin(phi), np.cos(phi), 0,  color='k', linestyle='dashed')  # Circle with z=0

# Overplot equator, front:
eq_front = np.linspace(0, np.pi, 100)
ax.plot( np.sin(eq_front), np.cos(eq_front), 0, color='k')  # Circle with z=0


# ### MERIDIAN ###
# Calculate vectors for meridian:
a = np.array([-np.sin(vpAlt), 0, np.cos(vpAlt)])
b = np.array([0, 1, 0])
b = b * np.cos(vpAz) + np.cross(a, b) * np.sin(vpAz) + a * np.dot(a, b) * (1 - np.cos(vpAz))

# Plot whole meridian, dashed:
ax.plot( a[0]*np.sin(phi) + b[0]*np.cos(phi),  b[1]*np.cos(phi),  a[2]*np.sin(phi) + b[2]*np.cos(phi),  color='k', linestyle='dashed')

# Overplot meridian, front:
meri_front = np.linspace(1/2*np.pi, 3/2*np.pi, 100)  # 1/2 pi - 3/2 pi
ax.plot( a[0]*np.sin(meri_front) + b[0]*np.cos(meri_front),  b[1]*np.cos(meri_front),  a[2]*np.sin(meri_front) + b[2]*np.cos(meri_front), color='k')



# ### FINISH PLOT ###
# Choose projection angle:
ax.view_init(elev=vpAlt*r2d, azim=0)
ax.axis('off')


# Force narrow margins:
pllim = r*0.6

ax.set_box_aspect([1,1,1])    # Was ax.set_aspect('equal'), no longer works:  Set axes to a 'square grid'

ax.set_xlim3d(-pllim,pllim)
ax.set_ylim3d(-pllim,pllim)
ax.set_zlim3d(-pllim,pllim)


# plt.show()
plt.tight_layout()
plt.savefig('sphere.png')
plt.close()


