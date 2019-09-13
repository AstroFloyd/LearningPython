#!/usr/bin/env python3

## @ file tz.py  Convert a date/time stamp between timezones

from datetime import datetime
from pytz import timezone, utc

nl = timezone('Europe/Amsterdam')
loc_dt = nl.localize(datetime(2015,12,21, 12,30,0))
print( loc_dt )

utc_dt = loc_dt.astimezone(utc)
print( utc_dt )

