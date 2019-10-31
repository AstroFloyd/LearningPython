#!/bin/env python3

import ephem

moons = ((ephem.Io(), 'i'),
         (ephem.Europa(), 'e'),
         (ephem.Ganymede(), 'g'),
         (ephem.Callisto(), 'c'))

# How to place discrete characters on a line that actually represents
# the real numbers -maxdist to +maxdist.

linelen = 100
maxdist = 40.


def put(line, character, dist):
    if abs(dist) > maxdist:
        return
    offset = dist / maxdist * (linelen - 1) / 2
    i = int(linelen / 2 + offset)
    line[i] = character


interval = ephem.hour * 3
now = ephem.now()
now -= now % interval

t = now - 1
while t < now + 10:
    line = [' '] * linelen
    put(line, 'J', 0)
    for moon, character in moons:
        moon.compute(t)
        put(line, character, moon.x)
    print(ephem.date(t).datetime(), ''.join(line).rstrip())
    t += interval

print('East is to the right;',
      ', '.join(['%s = %s' % (c, m.name) for m, c in moons]))
