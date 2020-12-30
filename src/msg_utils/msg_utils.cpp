
#include <unistd.h>
#include <assert.h>

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

FILE * log_file_mpi(const char *name, int nidx)
{
	char log_filename[512];
	snprintf(log_filename, 512, "%s.mpi_%d.log", name, nidx); 
	FILE *log_file = fopen(log_filename, "w+");
	assert(log_file != NULL);
	return log_file;
}

