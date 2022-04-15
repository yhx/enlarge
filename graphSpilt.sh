../build.sh release double

clear


# pattern cycle
file_name="../final.log"
rm final.log




neuron_num=42427
for split in 0 1 2 3 4 5
do
    dir_name="gs_cicle_"
    mkdir $dir_name$split
    cd $dir_name$split

    echo "pattern cycle=================================" $split
    echo "pattern cycle=================================" $split >> $file_name

    if [ "$split" == "5" ]; then
        split=0
    fi

    echo "spilt : $split"

    # rm -rf ./pattern_circle_iaf_mpi_*
    if [ ! -d pattern_circle_iaf_mpi_42427 ]; then
        echo "begin build..."
        ../../build/bin/pattern_circle_iaf_mpi $neuron_num 16 $split > ./tmp.log 
    fi
    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num >> $file_name

    # mpirun -n 16 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num >> $file_name


    echo "FINISH: " $split "!"
    cd ..
done

#fc
neuron_num=1240
pop_num=13
for split in 0 1 2 3 4 5
do
    dir_name="gs_fc_"
    mkdir $dir_name$split
    cd $dir_name$split

    echo "pattern fc======================================" $split
    echo "pattern fc======================================" $split >> $file_name

    if [ "$split" == "5" ]; then
        split=0
    fi

    echo "spilt : $split"

    if [ ! -d pattern_fc_iaf_mpi_1240_13 ]; then
        echo "begin build..."
        ../../build/bin/pattern_fc_iaf_mpi $neuron_num $pop_num 16 $split > ./tmp.log 
    fi
    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $pop_num >> $file_name

    # mpirun -n 16 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $pop_num >> $file_name

    echo "FINISH: " $split "!"
    cd ..
done



#forwd
neuron_num=1000
depth=2400
for split in 0 1 2 3 4 5
do
    dir_name="gs_forwd_"
    mkdir $dir_name$split
    cd $dir_name$split

    echo "pattern forwd======================================" $split
    echo "pattern forwd======================================" $split >> $file_name

    if [ "$split" == "5" ]; then
        split=0
    fi

    echo "spilt : $split"

    if [ ! -d pattern_forward_iaf_mpi_2400_1000 ]; then
        echo "begin build..."
        ../../build/bin/pattern_forward_iaf_mpi $depth $neuron_num 16 $split > ./tmp.log 
    fi

    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num >> $file_name

    # mpirun -n 16 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num >> $file_name

    echo "FINISH: " $split "!"
    cd ..
done



neuron_num=300
run_time=10.0
node_num=8
pop_num=9

for split in 0 1 2 3 4 5
do 
    dir_name="gs_random_"
    mkdir $dir_name$split
    cd $dir_name$split

    echo "pattern random======================================" $split
    echo "pattern random======================================" $split >> $file_name

    if [ "$split" == "5" ]; then
        split=0
    fi
    echo "spilt : $split"

    if [ ! -d pattern_random_iaf_mpi_"$neuron_num"_"$pop_num" ]; then
        echo "begin build... pattern_random_iaf_mpi_"$neuron_num"_"$pop_num" "
        ../../build/bin/pattern_random_iaf_mpi2echo "spilt : $split" $neuron_num $pop_num 16 $split > ./tmp.log 
    fi
    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_random_iaf_mpi_run $neuron_num $pop_num $run_time >> $file_name

    # mpirun -n 16 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_random_iaf_mpi_run $neuron_num $pop_num $run_time >> $file_name

    echo "FINISH: " $split "!"
    cd ..
done 