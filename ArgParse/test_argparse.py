#!/bin/env python3

"""Last example in https://docs.python.org/3/howto/argparse.html"""

# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete

parser = argparse.ArgumentParser(description="Calculate X to the power of Y.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

group  = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", help="produce verbose output")
group.add_argument("-q", "--quiet", action="store_true", help="produce little output")

parser.add_argument("x", type=int, help="the base")
parser.add_argument("y", type=int, help="the exponent")

argcomplete.autocomplete(parser)
args = parser.parse_args()

answer = args.x**args.y

if args.quiet:
    print(answer)
elif args.verbose:
    print("{} to the power {} equals {}".format(args.x, args.y, answer))
else:
    print("{}^{} == {}".format(args.x, args.y, answer))



