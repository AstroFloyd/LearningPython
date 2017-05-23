#!/bin/env python3
# coding: utf-8

import ephem
import numpy as np

utrecht = ephem.city('Utrecht')
#s = ephem.Sun(utrecht)
#sp = np.deg2rad((180.,90.-30.))
#print(s.az,s.alt, np.rad2deg((sp[0],sp[1])))
#
#ang = ephem.separation((s.az,s.alt),sp)
#print(np.rad2deg(ang), np.cos(ang), 1000*np.cos(ang))
#
#
#utrecht = ephem.city('Utrecht')
#utrecht.date = '2016/12/27 11:30:00'
#s = ephem.Sun(utrecht)
#print(np.rad2deg(s.az),np.rad2deg(s.alt))
#
#utrecht.date = '2016/12/27 12:30:00'
#s = ephem.Sun(utrecht)
#print(np.rad2deg(s.az),np.rad2deg(s.alt))


utrecht.date = ephem.Date((2016, 12, 27, 11, 30, 0.0))
s = ephem.Sun(utrecht)
print()
print(np.rad2deg(s.az),np.rad2deg(s.alt))
print(utrecht.date, ephem.localtime(utrecht.date))  # Note: first is in ephem format (y/m/d), the second in datetime format (y-m-d).  Convert datetime to ephem using ephem.Date()


print()
# winter:
utrecht.date = ephem.Date((2016, 12, 27, 11, 30, 0.0))
#print(utrecht.date, ephem.localtime(utrecht.date))  # Note: first is in ephem format (y/m/d), the second in datetime format (y-m-d).  Convert datetime to ephem using ephem.Date()
localtime = ephem.Date(ephem.localtime(utrecht.date))
tz = round((localtime - utrecht.date)*24)
print("winter:", utrecht.date, tz)

# summer:
utrecht.date = ephem.Date((2016, 6, 27, 11, 30, 0.0))
#print(utrecht.date, ephem.localtime(utrecht.date))  # Note: first is in ephem format (y/m/d), the second in datetime format (y-m-d).  Convert datetime to ephem using ephem.Date()
localtime = ephem.Date(ephem.localtime(utrecht.date))
tz = round((localtime - utrecht.date)*24)
print("summer:", utrecht.date, tz)


