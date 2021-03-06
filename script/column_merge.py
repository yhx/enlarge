#!/usr/bin/python

import os
import re
import sys, getopt

def column_merge(inputs=[], output=""):
    # print(inputs)
    # print(output)
    if (not inputs):
        return

    files = []
    outfile = open(output, "w+")
    for name in inputs:
        f = open(name, "r")
        files.append(f)

    f0 = files[0]
    files = files[1:]

    for l0 in f0:
        outfile.write(l0.strip("\n"))
        for f in files:
            line = f.readline()
            outfile.write(line.strip("\n"))
        outfile.write("\n")

def find_series_files(name):
    series_files = []
    fpath, fname_all = os.path.split(name)
    if fpath == "":
        fpath = "./"
    fname, fext = os.path.splitext(fname_all)
    pattern1 = fname + '[_\s]*\d+[_\s]*'+ fext + '$'

    # for _, _, files in os.walk(curr_dir):
    #     for f in files:
    #         if (re.match(pattern1, f)):
    #             series_files.append(f)

    for f in os.listdir(fpath):
        if (os.path.isfile(f) and re.match(pattern1, f)):
            series_files.append(os.path.join(fpath, f))

    while len(series_files) == 0:
        fname, fext2 = os.path.splitext(fname)
        if not fext2:
            print("Wrong file name to merge")
            exit(0)
        fext = fext2 + fext
        pattern2 = fname + '[_\s]*\d+[_\s]*'+ fext + '$'
        # for _, _, files in os.walk(curr_dir):
        #     for f in files:
        #         if (re.match(pattern2, f)):
        #             series_files.append(f)

        for f in os.listdir(fpath):
            if (os.path.isfile(f) and re.match(pattern2, f)):
                series_files.append(os.path.join(fpath, f))

    series_files.sort()
    out_file = os.path.join(fpath, fname + "_merge" + fext)

    return (series_files, out_file)

def main(argv):
    inputfile = []
    outputfile = ''

    usuage_msg = sys.argv[0] + '-i <inputfile1> -i <inputfile2> ... -i <inputfileN> -o <outputfile>  or ' + sys.argv[0] + ' -s <series filename without num>'

    try:
        opts, args = getopt.getopt(argv,"hi:o:s:",["ifile=","ofile=","sfile="])
    except getopt.GetoptError:
        print(usuage_msg)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usuage_msg)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile.append(arg)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--sfile"):
            inputfile, outputfile = find_series_files(arg)
    
    print('Column Merge: ' + ' '.join(str(e) for e in inputfile) + " to " + str(outputfile))
    column_merge(inputfile, outputfile);


if __name__ == "__main__":
    main(sys.argv[1:])
