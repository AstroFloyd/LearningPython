#!/usr/bin/env python3

import ephem
import datetime
import pytz

# Set observer's location:
obs = ephem.Observer()
obs.lat = '52.1:00'
obs.long = '5.1:00'

# Set moment of observation in LT:
tz = pytz.timezone('Europe/Amsterdam')
lt = tz.localize(datetime.datetime(2015,12,21, 12,30,0))
print("LT:  ",lt)

# Convert date/time to UTC and set as ephemeris time:
utc = lt.astimezone(pytz.utc)
print("UT:  ", utc)
obs.date = utc

# Compute the position of the Sun for the given location and moment:
sun = ephem.Sun(obs)
sun.compute(obs)

# Print settings and output:
print("Date: " + str(obs.date))
print(obs)
print("Sun alt: %6.2f°" % (sun.alt * 57.2957795))
print()




