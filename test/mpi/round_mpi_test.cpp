
#include "../../include/BSim.h"
#include <stdlib.h>
#include<time.h>

using namespace std;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int node_id = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

	const real run_time=1000e-3;


	time_t start,end;
	start=clock(); //time(NULL);
	MNSim mn("round_mpi_4_39999_39999_1_8");	//gpu
	mn.run(run_time);	

	end=clock(); //time(NULL);
	printf("exec time=%lf seconds\n",(double)(end-start) / CLOCKS_PER_SEC);
	return 0;
}
