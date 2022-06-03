#!/bin/bash

mpirun -np 16 --hostfile /archive/share/linhui2/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /archive/share/linhui2/pattern_network/spack_forward_run.sh 2400 1000 0
# mpirun -np 16 --hostfile /archive/share/linhui2/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /archive/share/linhui2/pattern_network/spack_forward_run.sh 2400 1 1

python ./sum.py 16

python compute_spikes.py /archive/share/linhui/new_bsim/bsim/forward_acc_test/rate_gpu_mpi_merge.IAF.log
