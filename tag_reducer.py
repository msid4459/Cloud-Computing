#!/usr/bin/python3

import sys


def read_map_output(file):
    """ Return an iterator for key, value pair extracted from file (sys.stdin).
    Input format:  key \t value
    Output format: (key, value)
    """
    for line in file:
        yield line.strip().split("\t", 1)


def tag_reducer():
    """ This mapper select tags and return the tag-owner information.

    Note: Unlike normal hadoop which provide the reducer with key and a list of the values
        e.g. tag1, (owner1, owner2...)
    , hadoop streaming instead provide the reducer with sorted (key, value) lines
        e.g. tag1, owner1
             tag1, owner2
             ...

    Furthermore, unlike normal hadoop which calls the reducer for every key,
    in hadoop streaming multiple keys maybe given to the reducer
        e.g. tag1, owner1
             tag1, owner2
             tag2, owner2
             tag2, owner3
             tag3, owner1

    Input format: tag \t owner
    Output format: tag \t {owner=count}
    """
    current_tag = ""
    owner_count = {}

    for tag, owner in read_map_output(sys.stdin):
        # Check if the tag read is the same as the tag currently being processed
        if current_tag != tag:

            # If this is the first line (indicated by the fact that current_tag will have the default value of "",
            # we do not need to output tag-owner count yet
            if current_tag != "":
                output = current_tag + "\t"

                for own, count in owner_count.items():
                    output += "{}={}, ".format(own, count)
                print(output.strip())

            # Reset the tag being processed and clear the owner_count dictionary for the new tag
            current_tag = tag
            owner_count = {}

        owner_count[owner] = owner_count.get(owner, 0) + 1

    # We need to output tag-owner count for the last tag. However, we only want to do this if the for loop is called.
    if current_tag != "":
        output = current_tag + "\t"
        for owner, count in owner_count.items():
            output += "{}={}, ".format(owner, count)
        print(output.strip())

if __name__ == "__main__":
    tag_reducer()
