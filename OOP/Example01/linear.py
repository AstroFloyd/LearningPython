""" Linear code, module file for second example from https://stackoverflow.com/q/1813117/1386750
    MvdS, 2020-10-20
"""


def token_of(line):
    return line.partition(':')[-1].strip()


def clean_input(filename):
    tokenList = open(filename, 'r')
    cleanedInput = []
    prevLine = 0

    for line in tokenList:
        if line.startswith('LINE:'):
            lineNo = int(token_of(line))
            diff = lineNo - prevLine - 1
            cleanedInput.append('\n' * (diff if diff>1 else diff+1))
            prevLine = lineNo
        else:
            cleanedLine = token_of(line)
            cleanedInput.append(cleanedLine + ' ')

    return cleanedInput

    
