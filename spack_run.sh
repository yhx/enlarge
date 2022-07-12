#!/bin/bash 
. /opt/spack/share/spack/setup-env.sh
spack load nccl@2.7.8-1%gcc@9.3.0
pwd
# cd /archive/share/yhx/bsim/part_data
# cuda-memcheck $*
$*
#./build/bin/forward_mpi_ml_test 3200 1000 1 8 4
#/archive/share/yhx/bsim/build/bin/lif_mpi
