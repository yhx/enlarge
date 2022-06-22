#!/bin/bash
 
mpirun -n 2 ./bin/iaf_mpi_hybrid_simulator
#  mpirun -n 4 --hostfile ../openmpi1.config -mca btl_tcp_if_include eno1 ../spack_run.sh ./bin/iaf_mpi_hybrid_simulator 
