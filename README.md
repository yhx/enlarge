# ENLARGE

ENLARGE is an Efficient SNN simuLation frAmewoRk on GPU clustErs .

The experiments are carried out on an eight-node cluster. 

Hardware details: 

1. GPU: 2x NVIDIA GTX 1080Ti, GRAM:2x 12GB. 
2. CPU: Intel E5-2680 V4. 
3. RAM: 512GB. 
4. Network: DELL S6000 switch, 10 Gb/s Ethernet. Software details: Ubuntu 20.04, CUDA 11.2, GCC 9.3.0, NVCC 11.1.74, CMAKE 3.16.3, Open MPI 4.0.3, and NEST V3.2 and V2.14.0.

Firstly, we need to put the project in the shared directory of the cluster and compile it with the following command (/enlarge is used to refer to the root directory of the ENLARGE project): /enlarge/build.sh release double

The experiments and test scripts used in the paper are as follows:

## 1. The comparison experiment with NEST

  All the following NEST-related experiments require the activation of the NEST environment:

```bash
source /nest-v3.2/bin/nest_vars.sh
```

  We also need to update the path of nest_var.sh and pattern\_{benchmark}\_3\_2.py in the spack\_{benchmark}\_run.sh scripts to their actual file paths ({benchmark} stands for the benchmarks, i.e., forward, circle, fc).

### 1.1 The Forward benchmark

  For NEST: 

```
mpirun -np 16 --hostfile /enlarge/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /enlarge/pattern_network/spack_forward_run.sh 2400 1000 0
```

  For ENLARGE: 

```
/enlarge/pattern_forward_acc.sh
```

### 1.2 The Circle benchmark

  For NEST: 

```
mpirun -np 16 --hostfile /enlarge/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /enlarge/pattern_network/spack_circle_run.sh 42427 0
```

  For ENLARGE: 

```
/enlarge/pattern_circle_acc.sh
```

1.3 The FC benchmark

  For NEST: 

```
mpirun -np 16 --hostfile /enlarge/pattern_network/openmpi.config -mca btl_tcp_if_include eno1 /enlarge/pattern_network/spack_fc_run.sh 1240 13 0
```

  For ENLARGE: 

```
/enlarge/pattern_fc_acc.sh
```

1.4 The MVC benchmark

  For NEST: 

```
python3 run_example_fullscale.py 0.2 0.117
```

  For ENLARGE:

(1) Download nest_network.tar.bz2 from  https://1drv.ms/u/s!Av2X6AViC6ZIyiQXIQNn1TAPOfrn?e=mm07zy and unzip it 

(2) Update the file path in /enlarge/multi-area-model/construct_network.cpp:

  Set FILE_NETWORK to the actual path of nest.network_0.20_0.117 file

  Set FILE_WEIGHT to the actual path of nest.weight_merge_0.20_0.117 file 

  Set FILE_POISSON to the actual path of nest.poisson_weight_0.20_0.117 file

(3) Recompile the ENLARGE project

(4) Build the network: 

```
/enlarge/build/bin/construct_network
```

(5) Run the experiment: 

```
mpirun -n 8 --hostfile ./openmpi1.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/run_network_multi_level
```

## 2. The experiment of scalability

  Command:

```
 /enlarge/run_scaling_all.sh
```

  The experiment results are saved in the run_all directory under the working directory. The filenames are forward_1.log\~forword_10.log, circle_1.log\~circle_10.log, fc_1.log\~fc_10.log (The experiments are performed ten times and the results are the average of them).

## 3. Optimization analysis

### 3.1 Multi-level architecture

  Command: 

```
/enlarge/run_multi_level.sh
```

  The experiment results are saved in the run_all directory under the working directory. The files forward_1.log\~forward_10.log, circle_1.log\~circle_10.log, and fc_1.log\~fc_10.log record the experiment results with multi-level architecture. The files multi_area_1.log\~multi_area_10.log record the experiment results of the MVC benchmark with multi-level architecture. The files multi_area2_1.log\~multi_area2_10.log record the experiment results of the MVC benchmark without multi-level architecture.

  To get other results:

(1)  Change the following codes in /enlarge/test/mpi/pattern\_{benchmark}\_iaf_mpi_run.cpp:

    MLSim mn(name, dt, thread_num);
    
    mn.run(run_time, thread_num, 1);  

  to:

    MNSim mn(name, dt);
    
    mn.run(run_time, 1);

(2) Modify the files named /enlarge/pattern\_{benchmark}.sh. Comment the lines marked as with multi-level and uncomment the lines marked as without multi-level

(3) Execute the command: 

```
/enlarge/run_multi_level.sh
```

  The results are recorded in the same files. i.e., forward_1.log\~forward_10.log, circle_1.log\~circle_10.log, and fc_1.log\~fc_10.log. 

### 3.2 Delay-ware spike delivery

  The results with delay-aware spike delivery are the same as the ENLARGE version experiment in experiment 1.4.

  The result without this optimization method:

  (1) Uncomment line 302 and comment 304\~331 in the file /enlarge/src/synapse/static/StaticData.kernel.cu

  (2) Run the command: 

    /build/bin/construct_network 
    
    mpirun -n 8 --hostfile ./openmpi1.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/run_network_multi_level

### 3.3 Batched communication procedure

  Run the command: /enlarge/run_delay_aware.sh

  The experiment results are saved in the run_all directory under the working directory. The filenames are multi_delay_1.log\~multi_delay_10.log.

### 3.4 The network partition algorithm

  (1) For ENLARGE project, check out the network_partition branch.

  (2) Create a new directory (referred to as DIR) and run 

```
/enlarge/network_partition.sh
```

  The execution time of the partition algorithm is printed in the terminal and the execution time of the SNN is logged in the file result.log.

  For METIS algorithm:

(3) Copy the *.graph files generated in (2) to the /enlarge/metis directory 

(4) Run 

```
/enlarge/metis/to_metis.sh *.graph 16
```

(5) Copy the *.metis file generated in (4) to the DIR/{benchmark}

(6) Under DIR, run 

```
/enlarge/network_partition.sh 
```



## 4. Experimental setup

ENLARGE runs on GPU clusters, and the single-node container images can be obtained through: docker pull yhx09/enlarge:v0.1

To set up the test environment using containers on a cluster (8 nodes, each with two graphic cards)

1.   Install Docker (>=20.10) (https://docs.docker.com/engine/install/) and NVIDIA Docker 2 (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on each node according to the official methods

2.   Set NVIDIA Docker as the default runtime. Add the following line to /etc/docker/dameon.json on each node: "default-runtime": "nvidia", (please refer to https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html)

3. Select a node as the management node, and execute the command: 

   ```
   docker swarm init --advertise-addr IP_ADDR  #(IP_ADDR is the IP address of the management node. The command returns the corresponding token of the network)
   ```

4. All other nodes are workers. On each worker node, execute the command: 

   ```
   docker swarm join -token TOKEN IP_ADDR:2377  #(TOKEN is the output of the previous command and the IP_ADDR is the IP address of the management node)
   ```

5. Create the network on the management node: 

   ```
   docker network create -d overlay --attachable enlarge_overlay
   ```

6. On every node execute the command: 

   ```
   docker pull yhx09/enlarge:v0.1  #(This image is built through /enlarge/Dockerfile)
   ```

7.   Modify the /enlarge/docker-compose.yml: update /archive/share/docker to the actual shared directory of the cluster

8. On the management node run: 

   ```
   docker stack deploy --compose-file /enlarge/docker-compose.yml enlarge
   ```

9. On the management node run: 

   ```
   docker service ps enlarge_worker  #(This command returns the IDs of all the Docker Service instances)
   ```

10. Iterates through all the Docker Service instances, and execute: 

    ```
    docker inspect -f='{{(index .NetworksAttachments 0).Addresses}}' SERVICE_ID #(The SERVICE_ID is the ID of the service instance and this command returns the IP address of each container)
    ```

11. On the management node run: 

    ```
    docker ps #(Assuming the ID of the container named enlarge_worker.XXX is CONTAINER_ID)
    ```

12. On the management node run: 

    ```
    docker exec -it CONTAINER_ID bash  #(Enter the container)
    ```

13.  Inside the container, establish public key based SSH authentication between all the containers (The IP address of the containers is that in step 10 and the default username/password is root/enlarge)

14.  Inside one container, copy /enlarge and /nest to the /share directory

15.  Carry out the experiments according to the previous section (We should add the -allow-run-as-root parameter to mpirun related commands or use a normal user account)