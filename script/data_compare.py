#!/usr/bin/python

import sys, getopt, os
import re
import ast
import numpy as np
import column_merge

def column_sub(file1="", file2=""):
    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)

    diff = data1 - data2

    print (np.max(np.abs(diff)))
    print (np.max(np.abs(diff))/max(np.max(np.abs(data1)),
                                    np.max(np.abs(data2))))
    return (np.max(np.abs(diff))) # , (np.max(np.abs(diff))/max(np.max(np.abs(data1)), np.max(np.abs(data2))))

def main(argv):
    file1 = ''
    file2 = ''

    usuage_msg = sys.argv[0] + ' -1 <file1> -2 <file2>'

    try:
        opts, args = getopt.getopt(argv,"h1:2:",["ifile=","ofile="])
    except getopt.GetoptError:
        print(usuage_msg)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usuage_msg)
            sys.exit()
        elif opt in ("-1", "--file1"):
            file1 = arg
        elif opt in ("-2", "--file2"):
            file2 = arg

    if not os.path.exists(file2):
        inputfile, outputfile = column_merge.find_series_files(file2)
        print('Column Merge: ' + ' '.join(str(e) for e in inputfile) + " to " + str(outputfile))
        column_merge.column_merge(inputfile, outputfile)
        diff = column_sub(file1, outputfile);
    else:
        diff = column_sub(file1, file2);

    return diff < 1e-5


if __name__ == "__main__":
    main(sys.argv[1:])

