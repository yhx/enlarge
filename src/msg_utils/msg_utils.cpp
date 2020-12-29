
#include <unistd.h>

#include "msg_utils.h"

int to_attach() 
{
	if (getenv("MPI_DEBUG") != NULL) {
		int id = -1;
		MPI_Comm_rank(MPI_COMM_WORLD, &id);
		volatile int i=0;
		printf("Process %d, id %d, is ready for attach\n", getpid(), id);
		fflush(stdout);
		while (i == 0) {
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return 0;
}

