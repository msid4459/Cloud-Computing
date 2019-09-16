#!/usr/bin/python3

import sys


def tag_mapper():
    """ This mapper select tags and return the tag-owner information.
    Input format: photo_id \t owner \t tags \t date_taken \t place_id \t accuracy
    Output format: tag \t owner
    """
    for line in sys.stdin:
        # Clean input and split it
        parts = line.strip().split("\t")

        # Check that the line is of the correct format
        # If line is malformed, we ignore the line and continue to the next line
        if len(parts) != 6:
          continue

        username = parts[1].strip()
        tags = parts[2].strip().split()

        for tag in tags:
            # In hadoop streaming, the output is sys.stdout, thus the print command
            print("{}\t{}".format(tag, username))

if __name__ == "__main__":
    tag_mapper()
