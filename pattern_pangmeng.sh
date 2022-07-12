clear

# to run:
# 1. create a directory like "mkdir test_pangmeng"
# 2. then "cd test_pangmeng" (program should run in a directory)
# 3. run "../pattern_pangmeng.sh"

# modify: 
# 1. just replace "50 100" to neuron numbers list needed to test

# all programs run with node number equals to 8 with MultiLevelSimulator

../build.sh prof double

# circle
node_num=8
for neuron_num in 50 100
do
    dir_name="circle_${node_num}_${neuron_num}"
    mkdir $dir_name
    cd $dir_name

    echo $node_num
    echo "neuron_num: " $neuron_num
    ../../build/bin/pattern_circle_iaf_mpi $neuron_num  $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num #>> $file_name

    # MultiNode version
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $cur_neuron_num #>> $file_name

    cd ..
done

# forward
node_num=8
depth=300
for neuron_num in 50 100
do
    dir_name="forward_${neuron_num}"
    mkdir $dir_name
    cd $dir_name

    echo "START: " $node_num "!"
    
    ../../build/bin/pattern_forward_iaf_mpi $(($depth * $node_num)) $neuron_num  $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num
    
    # multi node version
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num
    
    echo "FINISH: " $node_num "!"
    cd ..
done

# fc
node_num=8
for neuron_num in 50 100
do
    dir_name="fc_${neuron_num}_${node_num}"
    mkdir $dir_name
    cd $dir_name

    echo "START: " $node_num "!"
    echo "neuron num: " neuron_num]} "!"
    echo "Current population num: " $(($node_num + 5)) "!"
    
    ../../build/bin/pattern_fc_iaf_mpi $neuron_num $(($node_num + 5)) $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $(($node_num + 5))

    # MultiNodeVersion
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $cur_neuron_num $(($node_num + 5))
    
    echo "FINISH: " $node_num "!"
    cd ..
done
