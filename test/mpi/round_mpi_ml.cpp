
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
	const real run_time=1000e-3;

	if(argc !=5 && argc != 6)
	{
		if (node_id == 0) {
			printf("Need 4/5 paras. For example\n FR 1%%: %s n0 n1 fire_rate delay [algorithm]\n", argv[0]);
		}
		return 0;
	}

	const size_t n0=(size_t)atoi(argv[1]);
	const size_t n1=(size_t)atoi(argv[2]);

	const int fr = atoi(argv[3]);
	const int delay_step = atoi(argv[4]);

	SplitType split = SynapseBalance;

	char name[1024];

	if (argc == 6) {
		split = (SplitType)atoi(argv[5]);
		sprintf(name, "%s_%d_%ld_%ld_%d_%d_%d", "round_mpi", parts, n0, n1, fr, delay_step, split); 
	} else {
		sprintf(name, "%s_%d_%ld_%ld_%d_%d", "round_mpi", parts, n0, n1, fr, delay_step); 
	}

	time_t start,end;
	start=clock(); //time(NULL);
	MLSim mn(name, dt, 2);	//gpu
	mn.run(run_time, 2, true);	

	end=clock(); //time(NULL);
	printf("exec time=%lf seconds\n",(double)(end-start) / CLOCKS_PER_SEC);
	return 0;
}
