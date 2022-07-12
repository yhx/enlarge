#include <stdlib.h>
#include <time.h>
#include <fstream>
#include "../../include/BSim.h"

using namespace std;

const real dt = 1e-4;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int node_id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    if (argc != 5)
	{
		printf("Need 3 paras. For example\n FR 1%%: %s depth num_neuron run_time\n", argv[0]);
		return 0;
	}
	
	const int depth = atoi(argv[1]);  // 网络深度  
	const int N = atoi(argv[2]);  // 每层神经元的数量
    const real run_time = atoi(argv[3]);  // 运行时间
    const int choose = atoi(argv[4]);

    string file_name = "pattern_forward_iaf_mpi_" + to_string(depth) + "_" + to_string(N);
    const char * name = file_name.c_str();
    
    if (choose) {
        const int thread_num = 2;
        MLSim mn(name, dt, thread_num);	//gpu
        mn.run(run_time, thread_num, 2);	
    } else {
        MNSim mn(name, dt);
        mn.run(run_time, 2);
    }

    return 0;
}
