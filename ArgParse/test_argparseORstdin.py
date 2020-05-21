#!/bin/env python3

"""Last example in https://docs.python.org/3/howto/argparse.html"""

# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete
import fileinput

parser = argparse.ArgumentParser(description="Takes a message from cli or stdin.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--message", type=str, help="the message")

argcomplete.autocomplete(parser)
args = parser.parse_args()

if(args.message):
    message = args.message
else:
    print("stdin")
    message = ""
    for line in fileinput.input():
        message += line

# message = message.replace("\n"," ")
message = message[0:279]  # 280 characters max
print(message)

