#!/bin/env python3
# coding: utf-8

import ephem
import numpy as np
import datetime


utrecht = ephem.city('Utrecht')
print(utrecht)

# s = ephem.Sun(utrecht)
# sp = np.deg2rad((180.,90.-30.))
# print(s.az,s.alt, np.rad2deg((sp[0],sp[1])))
#
# ang = ephem.separation((s.az,s.alt),sp)
# print(np.rad2deg(ang), np.cos(ang), 1000*np.cos(ang))
#
#
# utrecht = ephem.city('Utrecht')
# utrecht.date = '2016/12/27 11:30:00'
# s = ephem.Sun(utrecht)
# print(np.rad2deg(s.az),np.rad2deg(s.alt))
#
# utrecht.date = '2016/12/27 12:30:00'
# s = ephem.Sun(utrecht)
# print(np.rad2deg(s.az),np.rad2deg(s.alt))

# Date and time: Python's datetime vs. ephem:
# utrecht.date = ephem.Date((2016, 12, 27, 11, 30, 0.0))
utrecht.date = ephem.Date((2016, 1, 2,  3, 4, 5.0))
print("\npyEphem vs. datetime:")
# Note: first is in ephem format (y/m/d), the second in datetime format
#  (y-m-d).  Convert datetime to ephem using ephem.Date()
print("                    utrecht.date (ephem format, UT):  ", utrecht.date)
print("ephem.localtime(utrecht.date) (datetime format, LT):  ",
      ephem.localtime(utrecht.date))

s = ephem.Sun(utrecht)
print("\nSun position: az: ", np.rad2deg(s.az), " -  alt: ",
      np.rad2deg(s.alt))

# TZ winter:
utrecht.date = ephem.Date((2016, 12, 27, 11, 30, 0.0))
# Note: first is in ephem format (y/m/d), the second in datetime format
#   (y-m-d).  Convert datetime to ephem using ephem.Date()
# print(utrecht.date, ephem.localtime(utrecht.date))
localtime = ephem.Date(ephem.localtime(utrecht.date))
tz = round((localtime - utrecht.date) * 24)
print("\nTZ winter:", utrecht.date, tz)

# TZ summer:
utrecht.date = ephem.Date((2016, 6, 27, 11, 30, 0.0))
# Note: first is in ephem format (y/m/d), the second in datetime format
#   (y-m-d).  Convert datetime to ephem using ephem.Date()
# print(utrecht.date, ephem.localtime(utrecht.date))
localtime = ephem.Date(ephem.localtime(utrecht.date))
tz = round((localtime - utrecht.date) * 24)
print("TZ summer:", utrecht.date, tz)


# Current time:
now = datetime.datetime.now()
print("\nNow in LT: ", now)

utrecht.date = ephem.date(now)
s = ephem.Sun(utrecht)
print("Sun position: NOTE: ephem assumes time is in UT!  az: ",
      np.rad2deg(s.az), " - alt: ", np.rad2deg(s.alt))

nowUT = datetime.datetime.utcnow()
print("\nNow in UT: ", nowUT)

nowUT = ephem.now()
print("Now in UT: ", nowUT)
print("Now in UT: ", nowUT.datetime())

utrecht.date = ephem.date(nowUT)
s = ephem.Sun(utrecht)
print("Sun position: use .utcnow:  az: ", np.rad2deg(s.az),
      " - alt: ", np.rad2deg(s.alt))


# Rise and set:
print("Rise time: ", ephem.localtime(s.rise_time))  # Note: in datetime format!
print("Set time:  ", ephem.localtime(s.set_time))   # Note: in datetime format!
