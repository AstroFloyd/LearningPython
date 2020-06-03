#!/bin/env python3

""" Main function, from https://stackoverflow.com/a/1813167/1386750
    MvdS, 2020-05-13
"""

import sys
import oop


def main():
    thefile = oop.FileParser(sys.argv[1])
    print(thefile.cleaned_input())

    
if __name__ == '__main__':
    main()

  
