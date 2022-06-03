#!/bin/bash

./clean.sh

mpirun -np 16 --hostfile /archive/share/linhui2/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /archive/share/linhui2/pattern_network/spack_fc_run.sh 3067 18 0

