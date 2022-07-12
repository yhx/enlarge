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

    if (argc != 4)
    {
		printf("Need 2 paras. For example\n FR 1%%: %s num_neuron run_time \n", argv[0]);
		return 0;
	}
	const size_t N = atoi(argv[1]);  // 每层神经元的数量
    const real run_time = atoi(argv[2]);
    const int choose = atoi(argv[3]);

    string file_name = "pattern_circle_iaf_mpi_" + to_string(N);
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
