clear

run_time=10.0

for delay_num in 1 2 4 8
do
    node_num=8
    neuron_num=42427
    dir_name="circle_${node_num}_${delay_num}"
    # dir_name="circle_${node_num}_${neuron_num}_${delay_num}"
    mkdir $dir_name
    cd $dir_name

    echo $node_num
    echo "current neuron_num: " $neuron_num
    echo "Current delay num: " $delay_num "!"

    # ../../build/bin/pattern_circle_iaf_mpi_delay $neuron_num $(($node_num * 2)) $delay_num > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num $run_time #>> $file_name

    # MultiNode version
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $cur_neuron_num #>> $file_name

    cd ..
done

depth=300
neuron_num=1000
for delay_num in 1 2 4 8
do
    node_num=8

    dir_name="forward_${node_num}_${delay_num}"
    mkdir $dir_name
    cd $dir_name

    echo "START: " $node_num "!"
    echo "Current delay num: " $delay_num "!"
    
    # ../../build/bin/pattern_forward_iaf_mpi_delay $(($depth * $node_num)) $neuron_num  $(($node_num * 2)) $delay_num > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num $run_time
    
    # multi node version
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num
    
    echo "FINISH: " $node_num "!"
    cd ..
done

for delay_num in 1 2 4 8
do
    node_num=8
    neuron_num=1240

    dir_name="fc_${node_num}_${delay_num}"
    mkdir $dir_name
    cd $dir_name

    echo "START: " $node_num "!"
    echo "Current neuron num: " $neuron_num "!"
    echo "Current population num: " $(($node_num + 5)) "!"
    echo "Current delay num: " $delay_num "!"
    
    # ../../build/bin/pattern_fc_iaf_mpi_delay $neuron_num $(($node_num + 5)) $(($node_num * 2)) $delay_num > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $(($node_num + 5)) $run_time

    # MultiNodeVersion
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $cur_neuron_num $(($node_num + 5))
    
    echo "FINISH: " $node_num "!"
    cd ..
done

