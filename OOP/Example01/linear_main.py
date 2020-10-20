#!/bin/env python3

""" Linear code, main file for second example from https://stackoverflow.com/q/1813117/1386750
    MvdS, 2020-10-20
"""

import sys
import linear


def main():
    cleanedInput = linear.clean_input(sys.argv[1])
    print(cleanedInput)

    
if __name__ == '__main__':
    main()

    
