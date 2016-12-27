# coding: utf-8
import ephem
utrecht = ephem.city('Utrecht')
s = ephem.Sun(utrecht)
sp = deg2rad((180.,90.-30.))
print(s.az,s.alt, rad2deg((sp[0],sp[1])))

ang = ephem.separation((s.az,s.alt),sp)
print(rad2deg(ang), cos(ang), 1000*cos(ang))

