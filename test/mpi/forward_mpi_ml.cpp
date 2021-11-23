#include "../../include/BSim.h"
#include <stdlib.h>
#include<time.h>

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int node_id = 0;
	int parts = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &parts);

	const real dt=1e-4;
	const real run_time=1.0;

	if(argc !=6 && argc != 7)
	{
		printf("Need 5/6 paras. For example\n: %s thread_num depth num_neuron fire_rate delay [algorithm]\n", argv[0]);
		return 0;
	}

	const int thread_num = atoi(argv[1]);
	const int depth=atoi(argv[2]);
	const int N=atoi(argv[3]);

	const int fr = atoi(argv[4]);
	const int delay_step = atoi(argv[5]);

	SplitType split = SynapseBalance;

	time_t start,end;
	start=clock(); //time(NULL);

	char name[1024];

	if (argc == 7) {
		split = (SplitType)atoi(argv[6]);
		sprintf(name, "%s_%d_%d_%d_%d_%d_%d", "standard_mpi", parts*thread_num, depth, N, fr, delay_step, split); 
	} else {
		sprintf(name, "%s_%d_%d_%d_%d_%d", "standard_mpi", parts*thread_num, depth, N, fr, delay_step); 
	}

	MLSim mn(name, dt, thread_num);	
	mn.run(run_time, thread_num, true);	//gpu

	end=clock(); //time(NULL);
	printf("exec time=%lf seconds\n",(double)(end-start) / CLOCKS_PER_SEC);
	return 0;
}
