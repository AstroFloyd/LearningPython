#!/bin/env python

# https://stackoverflow.com/a/32427177/1386750

import colored_traceback
colored_traceback.add_hook()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astroconst import pi,pi2, r2d,d2r
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Choose projection:
vpAlt = 20.0 * d2r
vpAz  = 15.0 * d2r

# Colours:
cStar  = '#FF0'  # (bright) yellow
cEarth = '#0B0'  # Green
cEq    = '#F00'  # Bright red
cEcl   = '#0FF'  # (bright) cyan


# Line styles:
lsFg = '--'  # Foreground: solid
lsBg = '--'  # Background: dashed

# Line widths:
lwArc = 3  # Arcs
lwArr = 2  # Arrows
lwFg  = 2  # Foreground
lwBg  = lwFg * 0.6  # Background

# Alphas:
bgfac   = 0.6  # Alpha facor bg/fg
aArr    = 1.0  # For arrows/arcs
aLine   = 0.7  # For lines
aLineBg = aLine*bgfac  # For background lines
aLbl    = 1.0  # For labels
aLblBg  = aLbl*bgfac  # For labels
aPlan   = 0.5  # Equatorial/ecliptic plane
aSphr   = 0.1  # Sphere
aEarth  = 1.0  # Earth ball

# z orders:
zPlan    = 10
zPlanlbl = 11
zSphr    = 20
zSphrlbl = 21
zEarth   = 80
zExt     = 90

eps = 23.439*d2r  # Obliquity of the ecliptic



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


def plot_line(az, xx,yy,zz, vpAz,vpAlt, lss, lws, clrs, alphas, zorders):
    
    # Use projection angles (bit vpAlt has opposite definition?) to find fg/bg:
    xx1,yy1,zz1 = rotate3d(xx,yy,zz, vpAz,-vpAlt)
    
    # Select the part of the line that is in the FOREGROUND:
    xx2 = xx[xx1>=0]
    yy2 = yy[xx1>=0]
    zz2 = zz[xx1>=0]
    
    if np.size(xx2)>1:  # 1, not 0, because of diff below
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
    
    if np.size(xx2)>1:  # 1, not 0, because of diff below
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


def plot_point(ax, xx,yy,zz, vpAz,vpAlt, sym, clrs, alphas, zorders):
    
    # Use projection angles (bit vpAlt has opposite definition?) to find fg/bg:
    xx1,yy1,zz1 = rotate3d(xx,yy,zz, vpAz,-vpAlt)
    
    if xx1>0:  # Plot using foreground attributes:
        ax.plot( xx, yy, zz, sym, color=clrs[0], alpha=alphas[0], zorder=zorders[0])
    else:  # Plot using background attributes:
        ax.plot( xx, yy, zz, sym, color=clrs[1], alpha=alphas[1], zorder=zorders[1])
    
    return


def plot_text(ax, xx,yy,zz, vpAz,vpAlt, text, size,weight, ha,va, clrs,alphas,zorders):
    
    # Use projection angles (bit vpAlt has opposite definition?) to find fg/bg:
    xx1,yy1,zz1 = rotate3d(xx,yy,zz, vpAz,-vpAlt)
    
    if xx1>0:  # Plot using foreground attributes:
        ax.text(xx,yy,zz, text, ha=ha, size=size, weight=weight, va=va, color=clrs[0], alpha=alphas[0], zorder=zorders[0])
    else:  # Plot using background attributes:
        ax.text(xx,yy,zz, text, ha=ha, size=size, weight=weight, va=va, color=clrs[1], alpha=alphas[1], zorder=zorders[1])
    
    return


def eq2ecl(ra,dec, eps):
    lon = np.arctan2( np.sin(ra)  * np.cos(eps) + np.tan(dec) * np.sin(eps),  np.cos(ra) ) % pi2
    lat =  np.arcsin( np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(eps) * np.sin(ra) )
    
    return lon,lat


def ecl2eq(lon,lat, eps):
    ra  = np.arctan2( np.sin(lon) * np.cos(eps)  -  np.tan(lat) * np.sin(eps),  np.cos(lon) ) % pi2
    dec =  np.arcsin( np.sin(lat) * np.cos(eps)  +  np.cos(lat) * np.sin(eps) * np.sin(lon) )
    
    return ra,dec


# Setup plot:
matplotlib.rcParams.update({'font.size': 20})  # Set font size for all text - default: 12
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)


# ### SPHERE ###
# Create a sphere:
phi    = np.linspace(0, 2*np.pi, 100)  # Azimuthal coordinate
phin   = np.linspace(0, 1, 100)        # Normalised array
zeros  = np.zeros(len(phi))            # Zeros for e.g. x/y/z plane
hphi1  = np.linspace(0,   np.pi,  50)  # Half of phi, for half a circle
hphi2  = hphi1 + pi                    # The other half 
hzeros = np.zeros(len(hphi1))          # Half a set of zeros
theta  = np.linspace(0,   np.pi, 100)  # Altitude coordinate

# Compute and plot sphere surface:
rSph = 1
xSph = np.outer(np.cos(phi), np.sin(theta))
ySph = np.outer(np.sin(phi), np.sin(theta))
zSph = np.outer(np.ones(np.size(phi)), np.cos(theta))
ax.plot_surface(xSph, ySph, zSph,  rstride=2, cstride=4, color='b', linewidth=0, alpha=aSphr, zorder=zSphr)


# ### EQUATOR 1/2 ###
# Compute equatorial plane:
xx = np.cos(hphi1)
yy = np.sin(hphi1)
zz = hzeros

# Plot equatorial plane:
plane_verts = [list(zip( xx, yy, zz ))]  # list() needed for zip() in Python3
plane = Poly3DCollection(plane_verts)
plane.set_facecolor('#F88')  # Reddish
plane.set_alpha(aPlan)
plane.set_zorder(zPlan)
ax.add_collection3d(plane)

# Plot equator (circle in the x-y plane):
plot_line(ax, xx,yy,zz, vpAz,vpAlt, [lsFg,lsBg], [lwFg,lwBg], [cEq,cEq], [aLine,aLineBg], [zSphrlbl,zPlanlbl])



# ### ECLIPTIC ###
# Create ecliptic (circle in the x-y plane, rotated about epsilon):
xx = np.cos(phi)  # cos RA
yy = np.sin(phi)  # sin RA
zz = zeros
xx,yy,zz = rot2d_x( xx, yy, zz, -eps)   # 1. Rotate 23° about the x-axis

# Plot ecliptic plane:
plane_verts = [list(zip( xx, yy, zz ))]  # list() needed for zip() in Python3
plane = Poly3DCollection(plane_verts)
plane.set_facecolor(cEcl)
plane.set_alpha(aLine)
plane.set_zorder(zPlan)
ax.add_collection3d(plane)

# Plot Ecliptic (circle in the x-y plane, rotated about epsilon):
plot_line(ax, xx,yy,zz, vpAz,vpAlt, [lsFg,lsBg], [lwFg,lwBg], [cEcl,cEcl], [aLine,aLineBg], [zSphrlbl,zPlanlbl])



# ### EQUATOR 2/2 ###
# Compute equatorial plane:
xx = np.cos(hphi2)
yy = np.sin(hphi2)
zz = hzeros

# Plot equatorial plane:
plane_verts = [list(zip( xx, yy, zz ))]  # list() needed for zip() in Python3
plane = Poly3DCollection(plane_verts)
plane.set_facecolor('#F88')  # Reddish
plane.set_alpha(aPlan)
plane.set_zorder(zPlan)
ax.add_collection3d(plane)

# Plot equator (circle in the x-y plane):
plot_line(ax, xx,yy,zz, vpAz,vpAlt, [lsFg,lsBg], [lwFg,lwBg], [cEq,cEq], [aLine,aLineBg], [zSphrlbl,zPlanlbl])


# Mark angle epsilon between equator and ecliptic:
rr = 0.17
xx = rr*zeros + 0.985    # Circle in y-z plane, around Aries (1,0,0)
yy = rr*np.cos(phin*eps)
zz = rr*np.sin(phin*eps)
plot_line(ax, xx,yy,zz, vpAz,vpAlt, ['-','-'], [lwFg*2,lwBg*2], [cEcl,cEcl], [aLine,aLineBg], [zSphrlbl,zPlanlbl])
plot_text(ax, xx[65],yy[65],zz[70], vpAz,vpAlt, r' $\varepsilon$  ', 'x-large', 'bold', 'left','center', [cEcl,cEcl], [aLbl,aLblBg], [zSphr,zSphr])



print(np.arccos(1-rr))



# ### MERIDIAN ###
# Plot meridian (circle in the x-z plane):
xx = np.cos(phi)
yy = zeros
zz = np.sin(phi)
plot_line(ax, xx,yy,zz, vpAz,vpAlt, [lsFg,lsBg], [lwFg,lwBg], ['k','k'], [aLine,aLineBg], [zSphrlbl,zPlanlbl])


# Plot Earth dot and label:
# ax.plot([0],[0],[0], 'o', color=cEart, alpha=aLine, zorder=zPlanlbl)
# Compute and plot sphere surface:
rSph = 0.05
ax.plot_surface(rSph*xSph, rSph*ySph, rSph*zSph,  rstride=2, cstride=4, color=cEarth, linewidth=0, alpha=aEarth, zorder=zEarth)
ax.text(0,0,0,  'Earth  ', ha='right', size='xx-large', weight='bold', va='center', color=cEarth, alpha=aLbl, zorder=zPlanlbl)

# Compute and plot equator (circle in the x-y plane):
xx = rSph*np.sin(hphi1-vpAz)  # Note: x/y swapped
yy = rSph*np.cos(hphi1-vpAz)
zz = hzeros
plot_line(ax, xx,yy,zz, vpAz,vpAlt, ['-',lsBg], [lwBg,lwBg], ['w','w'], [aLine,aLineBg], [zExt,zExt])



# x-axis: line observer - spring/aries point:
# ax.plot([0,2.5],[0,0], '--', color='k', alpha=aLine, zorder=zPlanlbl)
# ax.text(2.2,0, 0, 'x', ha='left', size='x-large', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)

ax.plot([-1,1],[0,0], '--', color='k', alpha=aLineBg, zorder=zPlanlbl)  # Intersection equatorial-ecliptical planes
ax.plot([1],[0],[0], 'o', color='k', alpha=aLine,   zorder=zExt)  # Plot dot north pole
ax.text(1,0,0, 'S', ha='left', size='xx-large', weight='bold', va='top', zorder=zExt)


# # y-axis: line observer - [0,1,0]:
# ax.plot([0,0],[0,2], '--', color='k', alpha=aLine, zorder=zPlanlbl)
# ax.text(0,1.3,0, 'y', ha='left', size='x-large', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)


# Plot north pole, rotation axis and labels 'NP','SP':
ax.plot([0,0],[0,0],[-1.5,1.5], '--', color='k', alpha=aLine, zorder=zPlanlbl)

ax.text(0,0, 1.1,  'NP ', ha='right', size='xx-large', weight='bold', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)
ax.plot([0],[0], [1], 'o', color='k', alpha=aLine,   zorder=zPlanlbl)  # Plot dot north pole

ax.text(0,0,-1., ' SP', ha='left', size='xx-large', weight='bold', va='center', color='k', alpha=aLblBg, zorder=-1)
ax.plot([0],[0],[-1], 'o', color='k', alpha=aLineBg, zorder=zPlanlbl)  # Plot dot south pole

# Cardinal points:
# ax.text(0,0, 1.1, ' z', ha='left', size='x-large', va='center', color='k', alpha=aLbl, zorder=zPlanlbl)




# STAR:
# Plot yellow star + label:
raSt =  45*d2r
decSt = 60*d2r
xst = np.cos(raSt)*np.cos(decSt)
yst = np.sin(raSt)*np.cos(decSt)
zst = np.sin(decSt)

plot_point(ax, xst,yst,zst, vpAz,vpAlt, 'o', [cStar,cStar], [aLbl,aLblBg], [zExt,zExt])
plot_text( ax, xst,yst,zst, vpAz,vpAlt, ' Star', 'xx-large','bold', 'left','bottom', [cStar,cStar], [aLbl,aLblBg], [zExt,zExt])



# Plot line observer - star:
plot_line(ax, np.array([0,xst]),np.array([0,yst]),np.array([0,zst]), vpAz,vpAlt, [':',':'], [lwFg,lwBg], [cStar,cStar], [aLine,aLineBg], [zSphrlbl,zPlanlbl])

# Plot equatorial 'meridian' (declination) star:
xx = np.cos(phin*decSt)
yy = zeros
zz = np.sin(phin*decSt)
xx,yy,zz = rot2d_z( xx, yy, zz, -raSt)   # 1. Rotate RA about the z-axis
plot_line(ax, xx,yy,zz, vpAz,vpAlt, ['-','-'], [lwFg*2,lwBg*2], [cEq,cEq], [aLine,aLineBg], [zSphrlbl,zPlanlbl])
plot_text(ax, xx[50],yy[50],zz[50], vpAz,vpAlt, r'$\delta$  ', 'x-large', 'bold', 'right','top', [cEq,cEq], [aLbl,aLblBg], [zSphr,zSphr])

# Line observer - foot equatorial 'meridian' star:
plot_line(ax, np.array([0,xx[0]]),np.array([0,yy[0]]),np.array([0,zz[0]]), vpAz,vpAlt, [':',':'], [lwFg,lwBg], [cEq,cEq], [aLine,aLineBg], [zSphrlbl,zPlanlbl])

# Plot arc 'right ascension' star on equator:
xx = np.cos(phin*raSt)
yy = np.sin(phin*raSt)
zz = zeros
plot_line(ax, xx,yy,zz, vpAz,vpAlt, ['-','-'], [lwFg*2,lwBg*2], [cEq,cEq], [aLine,aLineBg], [zSphrlbl,zPlanlbl])
plot_text(ax, xx[50],yy[50],zz[50]-0.02, vpAz,vpAlt, r'$\alpha$', 'x-large', 'bold', 'center','top', [cEq,cEq], [aLbl,aLblBg], [zSphr,zSphr])




# Star: ecliptic lines:
lSt,bSt = eq2ecl(raSt,decSt, eps)
raSt1,decSt1 = ecl2eq(lSt, 0, eps)  # (l=lSt,b=0)

# Plot 'ecliptic meridian' (latitude) star:
xx = np.cos(phin*bSt)
yy = zeros
zz = np.sin(phin*bSt)
xx,yy,zz = rot2d_z( xx, yy, zz, -lSt)   # 1. Rotate the longitude about the x-axis
xx,yy,zz = rot2d_x( xx, yy, zz, -eps)   # 1. Rotate over eps about the x-axis
plot_line(ax, xx,yy,zz, vpAz,vpAlt, ['-','-'], [lwFg*2,lwBg*2], [cEcl,cEcl], [aLine,aLineBg], [zSphrlbl,zPlanlbl])
plot_text(ax, xx[50],yy[50],zz[50], vpAz,vpAlt, r'$b$', 'x-large', 'bold', 'left','bottom', [cEcl,cEcl], [aLbl,aLblBg], [zSphr,zSphr])

# Plot line observer - 'ecliptic meridian' star:
plot_line(ax, np.array([0,xx[0]]), np.array([0,yy[0]]), np.array([0,zz[0]]), vpAz,vpAlt, [':',':'], [lwFg,lwBg], [cEcl,cEcl], [aLine,aLineBg], [zSphrlbl,zPlanlbl])

# Plot ecliptic longitude star:
xx = np.cos(phin*lSt)
yy = np.sin(phin*lSt)
zz = zeros
xx,yy,zz = rot2d_x( xx, yy, zz, -eps)   # 1. Rotate over eps about the x-axis
plot_line(ax, xx,yy,zz, vpAz,vpAlt, ['-','-'], [lwFg*2,lwBg*2], [cEcl,cEcl], [aLine,aLineBg], [zSphrlbl,zPlanlbl])
plot_text(ax, xx[50],yy[50],zz[50]-0.02, vpAz,vpAlt, r'$l$', 'x-large', 'bold', 'center','top', [cEcl,cEcl], [aLbl,aLblBg], [zSphr,zSphr])




# ### FINISH PLOT ###
# Choose projection angle:
ax.view_init(elev=vpAlt*r2d, azim=vpAz*r2d)
ax.axis('off')


# Force narrow margins:
pllim = 0.6  # For a unit sphere

ax.set_box_aspect([1,1,1])    # Was ax.set_aspect('equal'), no longer works:  Set axes to a 'square grid'

scale=0.9
ax.set_xlim3d(-pllim/scale,pllim/scale)
ax.set_ylim3d(-pllim/scale,pllim/scale)
ax.set_zlim3d(-pllim/scale,pllim/scale)


# plt.show()
plt.tight_layout()
plt.savefig('sphere.png')
plt.savefig('sphere.pdf')
plt.close()
print('Done')

