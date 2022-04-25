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
vpAlt = 20.0 * d2r
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


def rot2d_x(xx,yy,zz, theta):
    xx1 =  xx
    yy1 =  yy*np.cos(theta) + zz*np.sin(theta)
    zz1 = -yy*np.sin(theta) + zz*np.cos(theta)
    return xx1,yy1,zz1


def rot2d_y(xx,yy,zz, theta):
    xx1 =  xx*np.cos(theta) - zz*np.sin(theta)
    yy1 =  yy
    zz1 =  xx*np.sin(theta) + zz*np.cos(theta)
    return xx1,yy1,zz1


def rot2d_z(xx,yy,zz, theta):
    xx1 =  xx*np.cos(theta) + yy*np.sin(theta)
    yy1 = -xx*np.sin(theta) + yy*np.cos(theta)
    zz1 =  zz
    return xx1,yy1,zz1
    
    
def rotate3d(xx,yy,zz, vpAz,vpAlt):
    xx1,yy1,zz1 = rot2d_z( xx, yy, zz, vpAz)   # 1. Rotate vpAz about the z-axis
    xx2,yy2,zz2 = rot2d_y(xx1,yy1,zz1, vpAlt)  # 2. Rotate vpAlt about the new y-axis
    return xx2,yy2,zz2


def plot_line(xx,yy,zz, lss, lws, clrs, alphas, zorders):
    
    # Use projection angles (bit vpAlt has opposite definition?):
    xx1,yy1,zz1 = rotate3d(xx,yy,zz, vpAz,-vpAlt)
    
    # Select the part of the line that is in the FOREGROUND:
    xx2 = xx[xx1>0]
    yy2 = yy[xx1>0]
    zz2 = zz[xx1>0]
    
    # Find the largest 3D step made in the foreground.  Assume that this indicates the location of the jump:
    step2 = np.square(np.diff(xx2)) + np.square(np.diff(yy2)) + np.square(np.diff(zz2))  # Δr^2 = Δx^2 + Δy^2 + Δz^2
    smax  = np.amax(step2)    # Maximum step
    imax  = np.argmax(step2)  # Index where the maximum step is made
    # print(step2, imax)
    # print(imax, xx2[imax:imax+2])
    # print(imax, smax)
    
    if smax > 0.3:  # Glue the part after the jump to the part before the jump
        xx3 = np.hstack((xx2[imax+1:],xx2[0:imax+1]))
        yy3 = np.hstack((yy2[imax+1:],yy2[0:imax+1]))
        zz3 = np.hstack((zz2[imax+1:],zz2[0:imax+1]))
    else:  # No jump
        xx3 = xx2
        yy3 = yy2
        zz3 = zz2
    
    # Plot the part of the curve in the foreground:
    ax.plot( xx3, yy3, zz3, linestyle=lss[0], lw=lws[0], color=clrs[0], alpha=alphas[0], zorder=zorders[0])
    
    
    # Select the part of the line that is in the BACKGROUND:
    xx2 = xx[xx1<0]
    yy2 = yy[xx1<0]
    zz2 = zz[xx1<0]
    
    # Find the largest 3D step made in the background.  Assume it is the location of the jump:
    step2 = np.square(np.diff(xx2)) + np.square(np.diff(yy2)) + np.square(np.diff(zz2))  # Δr^2 = Δx^2 + Δy^2 + Δz^2
    smax  = np.amax(step2)    # Maximum step
    imax  = np.argmax(step2)  # Index where the maximum step is made.
    
    # Glue the part after the jump to the part before the jump:
    if smax > 0.3:
        xx3 = np.hstack((xx2[imax+1:],xx2[0:imax+1]))
        yy3 = np.hstack((yy2[imax+1:],yy2[0:imax+1]))
        zz3 = np.hstack((zz2[imax+1:],zz2[0:imax+1]))
    else:
        xx3 = xx2
        yy3 = yy2
        zz3 = zz2
        
    
    # Plot the part of the curve in the background:
    ax.plot( xx3, yy3, zz3, linestyle=lss[1], lw=lws[1], color=clrs[1], alpha=alphas[1], zorder=zorders[1])
    
    return



# Setup plot:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)


# ### SPHERE ###
# Create a sphere:
phi    = np.linspace(0, 2*np.pi, 100)  # Azimuthal coordinate
zeros  = np.zeros(len(phi))            # Zeros for e.g. x/y/z plane
hphi1  = np.linspace(0,   np.pi,  50)  # Half of phi, for half a circle
hphi2  = hphi1 + pi                    # The other half 
hzeros = np.zeros(len(hphi1))          # Half a set of zeros
theta  = np.linspace(0,   np.pi, 100)  # Altitude coordinate

# Compute and plot sphere surface:
rr = 1
xx = rr * np.outer(np.cos(phi), np.sin(theta))
yy = rr * np.outer(np.sin(phi), np.sin(theta))
zz = rr * np.outer(np.ones(np.size(phi)), np.cos(theta))

ax.plot_surface(xx, yy, zz,  rstride=2, cstride=4, color='b', linewidth=0, alpha=aSphr, zorder=zSphr)


# ### EQUATOR ###
# Plot equatorial plane:
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plane_verts = [list(zip(np.sin(phi), np.cos(phi), zeros ))]  # list() needed for zip() in Python3
plane = Poly3DCollection(plane_verts)
plane.set_facecolor('0.5')
plane.set_alpha(aPlan)  # No effect?  There is now...
plane.set_zorder(zPlan)
ax.add_collection3d(plane)

# Plot equator (circle in the x-y plane):
xx = np.sin(phi)
yy = np.cos(phi)
zz = zeros
plot_line(xx,yy,zz, [lsFg,lsBg], [lwFg,lwBg], ['r','r'], [aLine,aLineBg], [zSphrlbl,zPlanlbl])

# Plot equatorial plane:
eqpl = np.vstack([np.sin(phi-vpAz), np.cos(phi-vpAz)]).transpose()
poly = patch.Polygon(eqpl, alpha=0.5, edgecolor=None)


# ### MERIDIAN ###
# Plot meridian (circle in the x-z plane):
xx = np.sin(phi)
yy = zeros
zz = np.cos(phi)
plot_line(xx,yy,zz, [lsFg,lsBg], [lwFg,lwBg], ['k','k'], [aLine,aLineBg], [zSphrlbl,zPlanlbl])


# Plot Earth dot and label:
ax.plot([0],[0],[0], 'o', color='g', alpha=aLine, zorder=zPlanlbl)
ax.text(0,0,0,  'Earth ', ha='right', size='xx-large', weight='bold', va='center', color='g', alpha=aLbl, zorder=zPlanlbl)

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



# Plot Ecliptic (circle in the x-y plane, rotated about epsilon):
xx = np.sin(phi)
yy = np.cos(phi)
zz = zeros
xx,yy,zz = rot2d_x( xx, yy, zz, -23*d2r)   # 1. Rotate 23° about the x-axis
plot_line(xx,yy,zz, [lsFg,lsBg], [lwFg,lwBg], ['y','y'], [aLine,aLineBg], [zSphrlbl,zPlanlbl])










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

