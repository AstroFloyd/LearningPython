#!/bin/env python3
# coding: utf-8

import ephem
import numpy as np
import datetime


city = ephem.city('Tel Aviv')
print(city)

for yr in range(15):
    year = yr+25
    #print(year)
    date1 = ephem.Date((year, 1, 1,  0, 0, 0.0))
    #date1 = ephem.Date((-4713, 1, 1,  12, 0, 0.0))
    date2 = ephem.next_spring_equinox(date1)
    date3 = ephem.next_full_moon(date2)
    date4 = date3.datetime()
    
    weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    
    #print(date1)
    #print(date2)
    #print(date3)
    print(weekdays[date4.weekday()], date4, ephem.localtime(date3))

exit()

# s = ephem.Sun(city)
# sp = np.deg2rad((180.,90.-30.))
# print(s.az,s.alt, np.rad2deg((sp[0],sp[1])))
#
# ang = ephem.separation((s.az,s.alt),sp)
# print(np.rad2deg(ang), np.cos(ang), 1000*np.cos(ang))
#
#
# city = ephem.city('City')
# city.date = '2016/12/27 11:30:00'
# s = ephem.Sun(city)
# print(np.rad2deg(s.az),np.rad2deg(s.alt))
#
# city.date = '2016/12/27 12:30:00'
# s = ephem.Sun(city)
# print(np.rad2deg(s.az),np.rad2deg(s.alt))

# Date and time: Python's datetime vs. ephem:
# city.date = ephem.Date((2016, 12, 27, 11, 30, 0.0))
city.date = ephem.Date((2016, 1, 2,  3, 4, 5.0))
print("\npyEphem vs. datetime:")
# Note: first is in ephem format (y/m/d), the second in datetime format
#  (y-m-d).  Convert datetime to ephem using ephem.Date()
print("                    city.date (ephem format, UT):  ", city.date)
print("      city.date.datetime() (datetime format, UT):  ", city.date.datetime())
print("ephem.localtime(city.date) (datetime format, LT):  ",
      ephem.localtime(city.date))

s = ephem.Sun(city)
print("\nSun position: az: ", np.rad2deg(s.az), " -  alt: ",
      np.rad2deg(s.alt))

# TZ winter:
city.date = ephem.Date((2016, 12, 27, 11, 30, 0.0))
# Note: first is in ephem format (y/m/d), the second in datetime format
#   (y-m-d).  Convert datetime to ephem using ephem.Date()
# print(city.date, ephem.localtime(city.date))
localtime = ephem.Date(ephem.localtime(city.date))
tz = round((localtime - city.date) * 24)
print("\nTZ winter:", city.date, tz)

# TZ summer:
city.date = ephem.Date((2016, 6, 27, 11, 30, 0.0))
# Note: first is in ephem format (y/m/d), the second in datetime format
#   (y-m-d).  Convert datetime to ephem using ephem.Date()
# print(city.date, ephem.localtime(city.date))
localtime = ephem.Date(ephem.localtime(city.date))
tz = round((localtime - city.date) * 24)
print("TZ summer:", city.date, tz)


# Current time:
now = datetime.datetime.now()
print("\nNow in LT: ", now)

city.date = ephem.date(now)
s = ephem.Sun(city)
print("Sun position: NOTE: ephem assumes time is in UT!  az: ",
      np.rad2deg(s.az), " - alt: ", np.rad2deg(s.alt))

nowUT = datetime.datetime.utcnow()
print("\nNow in UT: ", nowUT)

nowUT = ephem.now()
print("Now in UT: ", nowUT)
print("Now in UT: ", nowUT.datetime())

city.date = ephem.date(nowUT)
s = ephem.Sun(city)
print("Sun position: use .utcnow:  az: ", np.rad2deg(s.az),
      " - alt: ", np.rad2deg(s.alt))


# Rise and set:
print("Rise time: ", ephem.localtime(s.rise_time))  # Note: in datetime format!
print("Set time:  ", ephem.localtime(s.set_time))   # Note: in datetime format!
