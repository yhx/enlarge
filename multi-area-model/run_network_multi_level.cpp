#include <stdlib.h>
#include <time.h>
#include <fstream>
#include "../include/BSim.h"

using namespace std;

const real dt=1e-4;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int node_id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    const real run_time = 10.0;
    const char * name = "multi_area_model_20_117";
    
    const int thread_num = 2;
    MLSim mn(name, dt, thread_num);	//gpu
	mn.run(run_time, thread_num, 1);	
    
    return 0;
}
