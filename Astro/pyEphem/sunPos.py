#!/usr/bin/env python3

import ephem
import datetime
import pytz

# Set observer's location:
obs = ephem.Observer()
obs.lat = '52.1:00'
obs.long = '5.1:00'


if False:
    # Set moment of observation in LT:
    tz = pytz.timezone('Europe/Amsterdam')
    lt = tz.localize(datetime.datetime(2015,12,21, 12,30,0))
    print("LT:  ",lt)
    
    # Convert date/time to UTC and set as ephemeris time:
    utc = lt.astimezone(pytz.utc)
    # utc = datetime.datetime(2015,12,21, 11,30,0)  # Manual utc
    print("UT:  ", utc)
    obs.date = utc
    
    # Compute the position of the Sun for the given location and moment:
    sun = ephem.Sun(obs)
    sun.compute(obs)
    
    # Print settings and output:
    print("Date: " + str(obs.date))
    print(obs)
    print("Sun az:  %6.2f°" % (sun.az * 57.2957795))
    print("Sun alt: %6.2f°" % (sun.alt * 57.2957795))
    print()


#print(ephem.localtime(utc))

import time as timer
#tw0 = timer.perf_counter()  # Wall time
tc0 = timer.process_time()  # CPU time

# 10^6x:
for year in range(2000,2010):
    for month in range(1,11):
        for day in range(1,11):
            for hour in range(10):
                for minute in range(10):
                    for second in range(10):
                        utc = datetime.datetime(2015,month,day, hour,minute,second)  # Manual utc
                        #print("UT:  ", utc)
                        
                        obs.date = utc
                        
                        # Compute the position of the Sun for the given location and moment:
                        sun = ephem.Sun(obs)
                        sun.compute(obs)
                        
                        # Print settings and output:
                        #print(selftr(obs.date), sun.az * 57.2957795, sun.alt * 57.2957795)
                        
#tw1 = timer.perf_counter()
tc1 = timer.process_time()

#print('Wall time:  %0.9f s' % (tw1-tw0))
#print('CPU time:   %0.9f s' % (tc1-tc0))
print(tc1-tc0, end=' ')
