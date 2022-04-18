#!/bin/env python

# https://stackoverflow.com/a/32427177/1386750

import colored_traceback
colored_traceback.add_hook()

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from astroconst import pi, r2d,d2r
import matplotlib.patches as patch

# Choose projection:
vpAlt = 15.0 * d2r
vpAz  = 30.0 * d2r

# Colours:
clrStar = '#FF0'

# Line styles:
lsFg = '-'   # Foreground: solid
lsBg = '--'  # Background: dashed

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
aSphr = 0.3  # Sphere

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
phi    = np.linspace(0, 2*np.pi, 100)  # Azimuthal coordinate
zeros  = np.zeros(len(phi))            # Zeros for e.g. x/y/z plane
hphi1  = np.linspace(0,   np.pi,  50)  # Half of phi, for half a circle
hphi2  = hphi1 + pi                    # The other half 
hzeros = np.zeros(len(hphi1))          # Half a set of zeros
theta  = np.linspace(0,   np.pi, 100)  # Altitude coordinate

xx = rr * np.outer(np.cos(phi), np.sin(theta))
yy = rr * np.outer(np.sin(phi), np.sin(theta))
zz = rr * np.outer(np.ones(np.size(phi)), np.cos(theta))

# Plot sphere surface:
ax.plot_surface(xx, yy, zz,  rstride=2, cstride=4, color='b', linewidth=0, alpha=aSphr, zorder=zSphr)


# ### EQUATOR ###
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ground_verts = [list(zip(np.sin(phi), np.cos(phi), zeros ))]  # list() needed for zip() in Python3
ground = Poly3DCollection(ground_verts)
ground.set_facecolor('0.5')
ground.set_alpha(aPlan)  # No effect?  There is now...
ground.set_zorder(zPlan)
ax.add_collection3d(ground)


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
# Plot meridian (circle at y=0; x-z plane): front half: solid, background half dashed:
ax.plot(np.sin(hphi1), hzeros, np.cos(hphi1), linestyle=lsFg, lw=lwFg, color='k', alpha=aLine,   zorder=zSphrlbl)
ax.plot(np.sin(hphi2), hzeros, np.cos(hphi2), linestyle=lsBg, lw=lwBg, color='k', alpha=aLineBg, zorder=zSphrlbl)








# Plot Earth dot and label:
ax.plot([0],[0],[0], 'o', color='w', alpha=aLine, zorder=zPlanlbl)
ax.text(0,0,0,  'Earth ', ha='right', size='xx-large', weight='bold', va='center', color='w', alpha=aLbl, zorder=zPlanlbl)

# # Plot meridian:
# meri_front = np.linspace(0, 1/2*np.pi, 100)  # 0 - 1/2 pi + vpAlt
# ax.plot( np.zeros(len(meri_front)), np.cos(meri_front), np.sin(meri_front), '--', color='k', alpha=aLine, zorder=zSphrlbl)

# x axis: line observer - foot meridian:
ax.plot([0,2.5],[0,0], '--', color='k', alpha=aLine, zorder=zPlanlbl)
ax.text(1,0,0, 'S', ha='left', size='xx-large', weight='bold', va='top', zorder=zPlanlbl)
ax.text(2.2,0, 0, 'x', ha='left', size='x-large', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)

# y axis: line observer - [0,1,0]:
ax.plot([0,0],[0,2], '--', color='k', alpha=aLine, zorder=zPlanlbl)
ax.text(0,1.3,0, 'y', ha='left', size='x-large', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)


# Plot north pole, line NP-observer-SP and labels 'NP','SP':
ax.plot([0],[0], [1], 'o', color='k', alpha=aLine,   zorder=zPlanlbl)  # Plot dot north pole
ax.plot([0],[0],[-1], 'o', color='k', alpha=aLineBg, zorder=zPlanlbl)  # Plot dot south pole
ax.plot([0,0],[0,0],[-1.5,1.5], '--', color='k', alpha=aLine, zorder=zPlanlbl)
ax.text(0,0, 1.1,  'NP ', ha='right', size='xx-large', weight='bold', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)
ax.text(0,0,-1.15, 'SP ', ha='right', size='xx-large', weight='bold', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)

# Cardinal points:
ax.text(0,0, 1.1, ' z', ha='left', size='x-large', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)




# ### FINISH PLOT ###
# Choose projection angle:
ax.view_init(elev=vpAlt*r2d, azim=vpAz*r2d)
ax.axis('off')


# Force narrow margins:
pllim = rr*0.6

ax.set_box_aspect([1,1,1])    # Was ax.set_aspect('equal'), no longer works:  Set axes to a 'square grid'

scale=0.9
ax.set_xlim3d(-pllim/scale,pllim/scale)
ax.set_ylim3d(-pllim/scale,pllim/scale)
ax.set_zlim3d(-pllim/scale,pllim/scale)


# plt.show()
plt.tight_layout()
plt.savefig('sphere.png')
plt.close()
print('Done')

