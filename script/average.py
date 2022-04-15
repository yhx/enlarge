#!/usr/bin/python

import sys, getopt
import re
import ast
import numpy as np

def line_sum(file1=""):
    data1 = np.loadtxt(file1)

    average = np.average(data1)
    print(average)



def main(argv):
    file1 = ''

    usuage_msg = sys.argv[0] + ' -f <file>'

    try:
        opts, args = getopt.getopt(argv,"hf:",["file="])
    except getopt.GetoptError:
        print(usuage_msg)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usuage_msg)
            sys.exit()
        elif opt in ("-f", "--file"):
            file1 = arg
    
    line_sum(file1);


if __name__ == "__main__":
    main(sys.argv[1:])

