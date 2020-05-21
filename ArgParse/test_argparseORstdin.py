#!/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import argparse
import argcomplete
import fileinput

parser = argparse.ArgumentParser(description="Takes a message from cli or stdin.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--message", type=str, help="the message")
parser.add_argument("-q", "--quiet",     action="store_true", help="do not produce any extra output")  # Default = False

argcomplete.autocomplete(parser)
args = parser.parse_args()

if(args.message):
    message = args.message
else:
    if(not args.quiet): print("stdin")
    message = ""
    for line in fileinput.input():  # Uses stdin if no file specified
        message += line

# message = message.replace("\n"," ")  # Remove newlines?
message = message[0:279]  # 280 characters max

if(not args.quiet): print("Message:  ", end="")
print(message)

