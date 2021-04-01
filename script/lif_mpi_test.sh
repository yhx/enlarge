#!/bin/bash
DIR=`dirname $0`
$DIR/../build/bin/lif_cpu
mpirun -n 4 $DIR/../build/bin/lif_mpi
python $DIR/data_compare.py -1 v.cpu.log -2 v.mpi.log
