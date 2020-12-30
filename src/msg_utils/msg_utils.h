
#ifndef MSG_UTILS
#define MSG_UTILS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"

int to_attach();

FILE * log_file_mpi(const char *name, int nidx);

template <typename T>
int print_mpi_x32(T *array, int size, const char *name)
{
	int id = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	int length = strlen(name)+35+33*size;

	char *log = (char*)malloc(sizeof(char)*length);
	memset(log, 0, sizeof(char)*length);
	int offset = sprintf(log, "%s %d:", name, id);
	for (int i=0; i<size; i++) {
		offset += sprintf(log+offset, "%d ", array[i]);
	}
	printf("%s\n", log);
	free(log);
	return offset;
}

#endif // MSG_UTILS

