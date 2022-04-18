#!/bin/env python

# https://stackoverflow.com/a/32427177/1386750

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from astroconst import pi, r2d,d2r
import matplotlib.patches as patch

# Choose projection:
vpAlt = 10.0 * d2r
vpAz  = -20.0 * d2r

# Line styles:
lsFg = '-'   # Foreground: solid
lsBg = '-'  # Background: dashed

# Line widths:
lwArc = 3  # Arcs
lwArr = 2  # Arrows
lwFg  = 2  # Foreground
lwBg  = lwFg * 0.6  # Background

# Alphas:
aArr  = 1.0  # For arrows/arcs
aLine = 0.7  # For lines
aLineBg = aLine*0.6  # For background lines
aLbl  = 1.0  # For labels
aPlan = 0.7  # Equatorial plane
aSphr = 0.5  # Sphere

# z orders:
zPlan    = 10
zPlanlbl = 11
zSphr    = 20
zSphrlbl = 21
zExt     = 90


# Setup plot:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)


# ### SPHERE ###
# Create a sphere:
rr = 1
phi   = np.linspace(0, 2*np.pi, 100)  # Azimuthal coordinate
hphi1 = np.linspace(0,   np.pi,  50)  # Half of phi, for half a circle
hphi2 = hphi1 + pi                    # The other half 
theta = np.linspace(0,   np.pi, 100)  # Altitude coordinate

xx = rr * np.outer(np.cos(phi), np.sin(theta))
yy = rr * np.outer(np.sin(phi), np.sin(theta))
zz = rr * np.outer(np.ones(np.size(phi)), np.cos(theta))

# Plot sphere surface:
ax.plot_surface(xx, yy, zz,  rstride=2, cstride=4, color='b', linewidth=0, alpha=aSphr, zorder=zSphr)


# ### EQUATOR ###
# Plot equator: front half solid, back half dashed:
ax.plot( np.sin(hphi1-vpAz), np.cos(hphi1-vpAz), 0, linestyle=lsFg, lw=lwFg, color='k', alpha=aLine,   zorder=zSphrlbl)  # Circle with z=0
ax.plot( np.sin(hphi2-vpAz), np.cos(hphi2-vpAz), 0, linestyle=lsBg, lw=lwBg, color='k', alpha=aLineBg, zorder=zSphrlbl)  # Circle with z=0

# Plot equatorial plane:
eqpl = np.vstack([np.sin(phi-vpAz), np.cos(phi-vpAz)]).transpose()
# print(np.shape(eqpl))
poly = patch.Polygon(eqpl, alpha=0.5, edgecolor=None)  # use transparency?
# plt.gca().add_patch(poly)
# ax.add_patch(poly)  # Same?


# ### MERIDIAN ###
# Plot meridian (circle at y=0): front half: solid, background half dashed:
ax.plot(np.sin(hphi1-vpAlt), np.zeros(len(hphi1)), np.cos(hphi1-vpAlt), linestyle=lsFg, lw=lwFg, color='k', alpha=aLine,   zorder=zSphrlbl)
ax.plot(np.sin(hphi2-vpAlt), np.zeros(len(hphi2)), np.cos(hphi2-vpAlt), linestyle=lsBg, lw=lwBg, color='k', alpha=aLineBg, zorder=zSphrlbl)



# ### FINISH PLOT ###
# Choose projection angle:
ax.view_init(elev=vpAlt*r2d, azim=vpAz*r2d)
ax.axis('off')


# Force narrow margins:
pllim = rr*0.6

ax.set_box_aspect([1,1,1])    # Was ax.set_aspect('equal'), no longer works:  Set axes to a 'square grid'

ax.set_xlim3d(-pllim,pllim)
ax.set_ylim3d(-pllim,pllim)
ax.set_zlim3d(-pllim,pllim)


# plt.show()
plt.tight_layout()
plt.savefig('sphere.png')
plt.close()
print('Done')

