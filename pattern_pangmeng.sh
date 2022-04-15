clear

# to run:
# 1. create a directory like "mkdir test_pangmeng"
# 2. then "cd test_pangmeng" (program should run in a directory)
# 3. run "../pattern_pangmeng.sh"

# modify: 
# 1. just replace "50 100" to neuron numbers list needed to test

# all programs run with node number equals to 8 with MultiLevelSimulator

# ../build.sh prof double

# circle
file_name="../log"
node_num=1
for neuron_num in 5304 10606 15910 21213 26516 31820 37123 42427 
do
    dir_name="circle_${node_num}_${neuron_num}"
    # mkdir $dir_name
    cd $dir_name

    # echo $node_num
    echo "======================================neuron_num: " $neuron_num
    echo "======================================neuron_num: " $neuron_num >> $file_name
    # ../../build/bin/pattern_circle_iaf_mpi $neuron_num  $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num >> $file_name

    # MultiNode version
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $cur_neuron_num #>> $file_name

    cd ..
done

# forward
node_num=8
depth=300
for neuron_num in 125 250 375 500 625 750 875 1000
do
    dir_name="forward_${neuron_num}"
    # mkdir $dir_name
    cd $dir_name

    echo "======================================neuron_num: " $neuron_num
    echo "======================================neuron_num: " $neuron_num >> $file_name
    
    # ../../build/bin/pattern_forward_iaf_mpi $(($depth * $node_num)) $neuron_num  $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num >> $file_name
    
    # multi node version
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num
    cd ..
done

# fc
node_num=8
for neuron_num in 155 310 465 620 775 930 1085 1240
do
    dir_name="fc_${neuron_num}_${node_num}"
    # mkdir $dir_name
    cd $dir_name

    echo "======================================neuron_num: " $neuron_num
    echo "======================================neuron_num: " $neuron_num >> $file_name
    # ../../build/bin/pattern_fc_iaf_mpi $neuron_num $(($node_num + 5)) $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $(($node_num + 5)) >> $file_name

    # MultiNodeVersion
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $cur_neuron_num $(($node_num + 5))
    
    cd ..
done
