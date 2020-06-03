#!/bin/env python3

""" Linear code, second example from https://stackoverflow.com/q/1813117/1386750
    MvdS, 2020-05-13
"""

import sys


def main():
    tokenList = open(sys.argv[1], 'r')
    cleanedInput = []
    prevLine = 0

    for line in tokenList:

        if line.startswith('LINE:'):
            lineNo = int(line.split(':', 1)[1].strip())
            diff = lineNo - prevLine - 1

            if diff == 0:
                cleanedInput.append('\n')
            if diff == 1:
                cleanedInput.append('\n\n')
            else:
                cleanedInput.append('\n' * diff)

            prevLine = lineNo
            continue

        cleanedLine = line.split(':', 1)[1].strip()
        cleanedInput.append(cleanedLine + ' ')

    print(cleanedInput)

    
if __name__ == '__main__':
    main()

    
