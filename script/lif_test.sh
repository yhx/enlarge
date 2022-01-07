#!/bin/bash
DIR=`dirname $0`
$DIR/../build/bin/lif_cpu
$DIR/../build/bin/lif_gpu
python $DIR/data_compare.py -1 v.cpu.log -2 v.gpu.log
