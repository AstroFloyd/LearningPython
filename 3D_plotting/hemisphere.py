#!/bin/env python

# https://stackoverflow.com/a/32427177/1386750

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from astroconst import pi, r2d,d2r


# Choose projection:
vpAlt =  15.0 * d2r
vpAz  = -20.0 * d2r
# vpAlt = 90.0 * d2r
# vpAz  = 0.0 * d2r

# Colours:
clrSun = '#FF0'

# Line widths:
lwArc = 3  # Arcs
lwArr = 2  # Arrows

# Alphas:
aArr  = 1.0  # For arrows/arcs
aLine = 0.7  # For lines
aLbl  = 1.0  # For labels
aGrnd = 0.7  # Ground
aDome = 0.3  # Dome

# z orders:
zGrnd    = 10
zGrndlbl = 11
zDome    = 20
zDomelbl = 21
zExt     = 90




# Setup plot:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',computed_zorder=False)

# ### HEMISPHERE ###
# Create a hemisphere/dome:
rad = 1
phi   = np.linspace(0, 2*np.pi, 100)  # Azimuthal coordinate
zeros = np.zeros(len(phi))            # X/Y/Z plane
hphi1 = np.linspace(0,   np.pi,  50)  # Half of phi, for half a circle
hphi2 = hphi1 + pi                    # The other half
theta = np.linspace(0, np.pi/2,  50)  # Altitude coordinate (0-90 only)

x = rad * np.outer(np.cos(phi), np.sin(theta))
y = rad * np.outer(np.sin(phi), np.sin(theta))
z = rad * np.outer(np.ones(np.size(phi)), np.cos(theta))

# Plot hemisphere/dome surface:
ax.plot_surface(x, y, z,  rstride=2, cstride=4, color='b', linewidth=0, alpha=aDome)

# ### HORIZON ###
# Plot whole horizon:
# ax.plot(np.sin(phi), np.cos(phi), 0,  color='g', alpha=1.0)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
verts = [list(zip(np.sin(phi), np.cos(phi), zeros ))]  # list() needed for zip() in Python3
poly = Poly3DCollection(verts)
poly.set_facecolor('g')
# poly.set_facecolor('#280')
poly.set_alpha(aGrnd)  # No effect?  There is now...
poly.set_zorder(zGrnd)
ax.add_collection3d(poly)

# Horizon line and label:
ax.plot(np.sin(hphi1+vpAz), np.cos(hphi1+vpAz), 0,  color='k', alpha=aLine, zorder=zDome)  # Foreground
ax.plot(np.sin(hphi2+vpAz), np.cos(hphi2+vpAz), 0,  '-', lw=0.8, color='k', alpha=aLine, zorder=zGrndlbl)  # Background
az = 90*d2r + vpAz
ax.text(np.sin(az),np.cos(az),0,  'horizon', ha='center', size='xx-large', weight='bold', va='top', color='k', alpha=aLbl, rotation=30, zorder=zGrndlbl)




# ### MERIDIAN ###
# Calculate vectors for meridian:
# a = np.array([-np.sin(vpAlt), 0, np.cos(vpAlt)])
# b = np.array([0, 1, 0])
# b = b * np.cos(vpAz) + np.cross(a, b) * np.sin(vpAz) + a * np.dot(a, b) * (1 - np.cos(vpAz))

# Plot whole (half) meridian, dashed:
# meri = np.linspace(0, np.pi, 100)  # 0 - pi - vpAlt
# ax.plot( a[0] * np.sin(meri) + b[0] * np.cos(meri),  b[1] * np.cos(meri),  a[2] * np.sin(meri) + b[2] * np.cos(meri), color='k', linestyle='dashed')

# Overplot meridian, front:
# meri_front = np.linspace(0, 1/2*np.pi, 100)  # 0 - 1/2 pi + vpAlt
# #ax.plot( a[0] * np.sin(meri_front) + b[0] * np.cos(meri_front),  b[1] * np.cos(meri_front),  a[2] * np.sin(meri_front) + b[2] * np.cos(meri_front), color='k')
# ax.plot( np.zeros(len(meri_front)), -np.cos(meri_front), np.sin(meri_front), color='k')


# Plot observer dot and label:
ax.plot([0],[0],[0], 'o', color='k', alpha=aLine, zorder=zGrndlbl)
ax.text(0,0,0,  'observer ', ha='right', size='xx-large', weight='bold', va='center', color='k', alpha=aLbl, zorder=zGrndlbl)

# Plot meridian:
meri_front = np.linspace(0, 1/2*np.pi, 100)  # 0 - 1/2 pi + vpAlt
ax.plot( np.zeros(len(meri_front)), np.cos(meri_front), np.sin(meri_front), '--', color='k', alpha=aLine, zorder=zDomelbl)

# Line observer - foot meridian (S):
ax.plot([0,0],[0,2],[0,0], '--', color='k', alpha=aLine, zorder=zGrndlbl)
ax.text(0,1,0, 'S', ha='left', size='xx-large', weight='bold', va='top', zorder=zGrndlbl)


# Plot zenith, line observer - zenith and label 'Zenith':
ax.plot([0],[0],[1], 'o', color='k', alpha=aLine, zorder=zGrndlbl)  # Plot dot zenith
ax.plot([0,0],[0,0],[0,1], '--', color='k', alpha=aLine, zorder=zGrndlbl)
ax.text(0,0,1.1,  'zenith', ha='center', size='xx-large', weight='bold', va='center', color='k', alpha=aLbl, zorder=zGrndlbl)



# Plot star:
az =  45*d2r
alt = 45*d2r

# Plot line observer - star:
ax.plot([0,np.sin(az)*np.cos(alt)],[0,np.cos(az)*np.cos(alt)],[0,np.sin(alt)], '--', color='k', alpha=aLine, zorder=zGrndlbl)

# Plot 'meridian' star:
meri_front = np.linspace(0, 1/2*np.pi, 100)  # 0 - 1/2 pi + vpAlt
ax.plot( np.sin(az)*np.cos(meri_front), np.cos(az)*np.cos(meri_front), np.sin(meri_front), '--', color='k', alpha=aLine, zorder=zGrndlbl)

# Line observer - foot 'meridian' star:
ax.plot([0,np.sin(az)],[0,np.cos(az)],[0,0], '--', color='k', alpha=aLine, zorder=zGrndlbl)

# Plot arc 'azimuth' star on horizon:
arr = np.linspace(0, az, 100)  # 0 - 1/2 pi + vpAlt
ax.plot(np.sin(arr), np.cos(arr), 0,  color='k', alpha=aArr, lw=lwArc, zorder=zDomelbl)

# Plot arrow 'azimuth' angle and add label 'A':
rr = 0.4
ax.plot(rr*np.sin(arr), rr*np.cos(arr), rr*0,  color='k', alpha=aArr, lw=lwArr, zorder=zGrndlbl)
ax.text( rr*np.sin(az/2), rr*np.cos(az/2), rr*0, '  A', ha='left', size='large', va='top', color='k', alpha=aArr, zorder=zGrndlbl)




# Plot arc 'altitude' star on dome:
meri_front = np.linspace(0, alt, 100)  # 0 - 1/2 pi + vpAlt
ax.plot( np.sin(az)*np.cos(meri_front), np.cos(az)*np.cos(meri_front), np.sin(meri_front), '-', color='k', alpha=aArr, lw=lwArc, zorder=zDomelbl)

# Plot arrow 'altitude' angle and add label 'h':
rr = 0.2
ax.plot( rr*np.sin(az)*np.cos(meri_front), rr*np.cos(az)*np.cos(meri_front), rr*np.sin(meri_front), '-', color='k', alpha=aArr, lw=lwArr, zorder=zGrndlbl)
ax.text( rr*np.sin(az)*np.cos(alt/2), rr*np.cos(az)*np.cos(alt/2), rr*np.sin(alt/2), ' h', ha='left', size='large', va='center', color='k', alpha=aArr, zorder=zGrndlbl)


# Plot yellow star + label:
ax.plot([np.sin(az)*np.cos(alt)],[np.cos(az)*np.cos(alt)],[np.sin(alt)], 'o', color=clrSun, alpha=1.0, zorder=zExt)
ax.text(np.sin(az)*np.cos(alt), np.cos(az)*np.cos(alt), np.sin(alt), ' Sun', ha='left', size='xx-large', weight='bold', va='top', color=clrSun, alpha=1.0, zorder=zExt)
# print(meri_front*r2d, vpAlt*r2d)







# ### FINISH PLOT ###
# Choose projection angle:
ax.view_init(elev=vpAlt*r2d, azim=-vpAz*r2d)
ax.axis('off')


# Force narrow margins:
pllim = rad*0.6
ax.set_box_aspect([1,1,1])    # Was ax.set_aspect('equal'), no longer works:  Set axes to a 'square grid'

scale=1.0
ax.set_xlim3d(-pllim/scale,pllim/scale)
ax.set_ylim3d(-pllim/scale,pllim/scale)
ax.set_zlim3d(-0.5*pllim/scale,1.5*pllim/scale)


plt.tight_layout()
plt.savefig('hemisphere.png')
# plt.show()
plt.close()
print('Done')


