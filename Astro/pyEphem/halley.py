#!/bin/env python3

import ephem
from scipy.optimize import fmin


def separation(date):
    athens.date = ephem.Date(date)
    m = ephem.Moon(athens)
    a = ephem.star('Aldebaran')
    a.compute(athens)
    sep = ephem.separation(m, a)/ephem.degree
    return sep


def relsepminusone(date):
    sep = separation(date)
    rad = m.radius/ephem.degree
    rsep = sep/rad
    return rsep-1


athens = ephem.city('Athens')

athens.date = ephem.Date((509, 3, 11, 13, 37, 37.719))  # UT

m = ephem.Moon(athens)
print("Moon rise: ", m.rise_time, "UT")
print("Moon set:  ", m.set_time,  "UT")

print("Moon altitude: ", m.alt/ephem.degree)
print("Moon azimuth:  ", m.az/ephem.degree)


s = ephem.Sun(athens)
print()
print("Sun rise: ", s.rise_time, "UT")
print("Sun set:  ", s.set_time,  "UT")

print("Sun altitude: ", s.alt/ephem.degree)
print("Sun azimuth:  ", s.az/ephem.degree)


a = ephem.star('Aldebaran')
#a = ephem.FixedBody(epoch=509)
#a._ra = 1 #ephem.hours(3.20754*15) #*15*ephem.degree  # ephem.hours('02:04:10.278')   # in hours
#a._dec = 12.18701*ephem.degree    # ephem.degrees('+46:41:16.21') # in degrees
#a._a_epoch = ephem.Date((509, 3, 11,  13, 0, 0.0))
a.compute(athens)

print()
print("Aldebaran rise: ", a.rise_time, "UT")
print("Aldebaran set:  ", a.set_time,  "UT")

print("Aldebaran altitude: ", a.alt/ephem.degree)
print("Aldebaran azimuth:  ", a.az/ephem.degree)


print()
print("Separation: ", ephem.separation(m, a)/ephem.degree)

startTime = ephem.Date((509, 3, 11,  13, 0, 0.0))  # UT
endTime = startTime + 1.5/24  # in days
time = startTime
while time < endTime:
    # athens.date = ephem.Date(time)
    # m = ephem.Moon(athens)
    # a = ephem.star('Aldebaran')
    # a.compute(athens)
    # sep = ephem.separation(m, a)/ephem.degree
    sep = separation(time)
    rad = m.radius/ephem.degree

    # print("Time, Separation: ", ephem.date(time), sep, sep/rad,
    #      relsepminusone(time))
    time += 2/1440  # In days

print()

start = ephem.newton(relsepminusone, ephem.Date((509, 3, 11,  13, 00, 0.0)),
                     ephem.Date((509, 3, 11,  13, 30, 0.0)))
end = ephem.newton(relsepminusone, ephem.Date((509, 3, 11,  13, 30, 0.0)),
                     ephem.Date((509, 3, 11,  14, 00, 0.0)))
print("Start occultation: ", ephem.date(start))
print("End occultation:   ", ephem.date(end))

minDist = fmin(relsepminusone, ephem.Date((509, 3, 11,  13, 30, 0.0)), disp=0)  # disp=0: quiet
print("Minimum approach:  ", ephem.Date(minDist))

print(a.ra/ephem.degree/15, a.dec/ephem.degree)


athens.date = ephem.date((2000,6,6))
a.compute(athens)
print(a.ra/ephem.degree/15, a.dec/ephem.degree)
