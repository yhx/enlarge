#!/usr/bin/python

import os
import sys, getopt
import re
import ast
import column_merge
import numpy as np

def np_array_sub(file1, file2):
    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)

    diff = data1 - data2

    np.savetxt("diff.res", diff)

    delta = np.max(np.abs(diff))

    if delta>1e-7:
        print ("Diff")
    else:
        print ("Same")

def column_sub(file1="", file2=""):
    infile1 = open(file1, "r+")
    infile2 = open(file2, "r+")
    outfile = open("diff.res", "w+")

    inlines1 = infile1.readlines()
    inlines2 = infile2.readlines()

    diffed = False

    linenum = 0;
    for l0,l1 in zip(inlines1, inlines2):
        line0 = l0.strip()
        line1 = l1.strip()
        string0 = filter(None, re.split(r'\s*[;,:\s]\s*', line0))
        string1 = filter(None, re.split(r'\s*[;,:\s]\s*', line1))
        data0 = list(map(ast.literal_eval, string0))
        data1 = list(map(ast.literal_eval, string1))

        assert len(data0)==len(data1)

        outfile.write('Line ' + str(linenum) + ' \t')
        columnum = 0
        for i,j in zip(data0, data1):
            diff = i-j
            if abs(diff) > 1e-7:
                diffed = True
                outfile.write(str(columnum) + ':' + str(diff) + ' \t');
            columnum = columnum + 1
        outfile.write('\n')
        linenum = linenum + 1

    if diffed:
        print ("Diff")
    else:
        print ("Same")



def main(argv):
    file1 = ''
    file2 = ''
    files = ''

    usuage_msg = sys.argv[0] + ' -1 <file1> -2 <file2> or ' + sys.argv[0] + ' -s <series filename without series num>'

    try:
        opts, args = getopt.getopt(argv,"h1:2:s:",["file1=","file2=", "files="])
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
        elif opt in ("-s", "--files"):
            files = arg
    if file1 == '' or file2 == '':
        inputfile, outputfile = column_merge.find_series_files(files)
        print('Column Merge: ' + ' '.join(str(e) for e in inputfile) + " to " + str(outputfile))
        column_merge.column_merge(inputfile, outputfile)
        np_array_sub(files, outputfile)

    elif not os.path.exists(file2):
        inputfile, outputfile = column_merge.find_series_files(file2)
        print('Column Merge: ' + ' '.join(str(e) for e in inputfile) + " to " + str(outputfile))
        column_merge.column_merge(inputfile, outputfile)
        np_array_sub(file1, outputfile);
    else:
        np_array_sub(file1, file2);


if __name__ == "__main__":
    main(sys.argv[1:])

