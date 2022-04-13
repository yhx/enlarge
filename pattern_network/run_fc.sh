#!/bin/bash

./clean.sh

mpirun -np 16 --hostfile /archive/share/linhui2/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /archive/share/linhui2/pattern_network/spack_fc_run.sh 1240 13 0
# mpirun -np 16 --hostfile /archive/share/linhui2/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /archive/share/linhui2/pattern_network/spack_fc_run.sh 10 13 1
# mpirun -np 16 --hostfile /archive/share/linhui2/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /archive/share/linhui2/pattern_network/spack_fc_run.sh 3067 18 0

python ./sum.py 16

python compute_spikes.py /archive/share/linhui/new_bsim/bsim/fc_acc_test/rate_gpu_mpi_merge.IAF.log
