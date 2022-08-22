#!/bin/env python3

# https://stackoverflow.com/a/47663599/1386750

import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import datetime as dt
import time


loc = coord.EarthLocation(lon=5.0 * u.deg,
                          lat=52.0 * u.deg)

tm = Time.now()

tw0 = time.perf_counter()  # Wall time
tc0 = time.process_time()  # CPU time

tm = Time('2019-10-31 07:40:28.212')
# print(tm, tm.jd)

sun = coord.get_sun(tm)                        # Compute Sun positin (in geocentric ra, dec, dist)
altaz = coord.AltAz(location=loc, obstime=tm)  # Set parameters for coodinate transformation?
sunAltAz = sun.transform_to(altaz)             # Coordinate transformation
print(sun)
print(sunAltAz)
print(sunAltAz.alt, sunAltAz.az)


tw1 = time.perf_counter()
tc1 = time.process_time()

print('Wall time: ', tw1-tw0, 's')
print('CPU time:  ', tc1-tc0, 's')

# First calculation above takes ~30x more time (~1.1s) than the following calculations (~0.038 s)
minute = 0
for hour in range(24):
# for hour in range(10):  # 100x for timing
#     for minute in range(10):
    tm = Time(dt.datetime(2019,10,31,  hour,minute,0))
    sun = coord.get_sun(tm)
    altaz = coord.AltAz(location=loc, obstime=tm)  # Must be computed every time - sets the parameters for the
    #                                                coordinate transformation?
    sunAltAz = sun.transform_to(altaz)
    print(tm, sunAltAz.alt, sunAltAz.az)

tw2 = time.perf_counter()
tc2 = time.process_time()

print('Wall time: ', tw2-tw1, 's')
print('CPU time:  ', tc2-tc1, 's')
# 100x in ~4s (or 7s, when a bad core is hit?) ~ 0.04s/calc / 25 calls/s.  SolTrack.py: 12340 calls/s, 494x faster!
