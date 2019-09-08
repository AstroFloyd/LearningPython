#!/bin/env python3

# https://stackoverflow.com/a/32427177/1386750

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r2d = 180 / np.pi
d2r = np.pi / 180

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a sphere:
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
r = 1

x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Choose projection:
elev = 10.0 * d2r
rot  = 80.0 * d2r

ax.plot_surface(x, y, z,  rstride=2, cstride=4, color='b', linewidth=0, alpha=0.5)


# Calculate vectors for longitude circle:
a = np.array([-np.sin(elev), 0, np.cos(elev)])
b = np.array([0, 1, 0])
b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))

ax.plot(np.sin(u), np.cos(u), 0,  color='k', linestyle='dashed')

horiz_front = np.linspace(0, np.pi, 100)
ax.plot(np.sin(horiz_front),np.cos(horiz_front),0,color='k')

vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u),color='k', linestyle = 'dashed')
ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front), b[1] * np.cos(vert_front), a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front),color='k')

ax.view_init(elev = elev * r2d, azim = 0)
ax.axis('off')

# Force narrow margins:
pllim = r*0.6
ax.set_xlim3d(-pllim,pllim)
ax.set_ylim3d(-pllim,pllim)
ax.set_zlim3d(-pllim,pllim)
ax.set_aspect('equal')

#plt.show()
plt.tight_layout()
plt.savefig('sphere.png')
plt.close()


