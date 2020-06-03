"""Example class - https://stackoverflow.com/a/1813167/1386750"""


def token_of(line):
    return line.partition(':')[-1].strip()


class FileParser(object):
    def __init__(self, filename):
        self.tokenList = open(filename, 'r')

    def cleaned_input(self):
        cleanedInput = []
        prevLine = 0

        for line in self.tokenList:
            if line.startswith('LINE:'):
                lineNo = int(token_of(line))
                diff = lineNo - prevLine - 1
                cleanedInput.append('\n' * (diff if diff>1 else diff+1))
                prevLine = lineNo
            else:
                cleanedLine = token_of(line)
                cleanedInput.append(cleanedLine + ' ')

        return cleanedInput


