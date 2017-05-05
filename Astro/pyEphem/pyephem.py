#!/bin/env python3
# coding: utf-8

import ephem
import numpy as np

utrecht = ephem.city('Utrecht')
s = ephem.Sun(utrecht)
sp = np.deg2rad((180.,90.-30.))
print(s.az,s.alt, np.rad2deg((sp[0],sp[1])))

ang = ephem.separation((s.az,s.alt),sp)
print(np.rad2deg(ang), np.cos(ang), 1000*np.cos(ang))

