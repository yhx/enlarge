#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pthread.h>

int proc_num;
int proc_rank;
int proc_datasize = 64;

/**
 * @brief Illustrates how to initialise the MPI environment with multithreading
 * support and ask for the MPI_THREAD_MULTIPLE level.
 * @details This application initialised MPI and asks for the 
 * MPI_THREAD_MULTIPLE thread support level. It then compares it with the
 * thread support level provided by the MPI implementation.
 **/
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    int *send = (int *)malloc(sizeof(int) * proc_datasize);
    for (int i = 0; i < proc_datasize; ++i) {
        send[i] = i;
    }

    
    int *recv = (int *)malloc(sizeof(int) * proc_datasize * proc_num);
    MPI_Barrier(MPI_COMM_WORLD);
    double ts = MPI_Wtime();
    for (int i = 0; i < 100000; ++i) {
        MPI_Alltoall(send, proc_datasize, MPI_INT, recv, proc_datasize, MPI_INT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double te = MPI_Wtime();
    printf("1: Proc %d one send time: %lf\n", proc_rank, te - ts);
    MPI_Barrier(MPI_COMM_WORLD);

    ts = MPI_Wtime();
    for (int i = 0; i < 100000; ++i) {
        MPI_Alltoall(send, proc_datasize / 2, MPI_INT, recv, proc_datasize / 2, MPI_INT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Alltoall(send, proc_datasize / 2, MPI_INT, recv, proc_datasize / 2, MPI_INT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    te = MPI_Wtime();
    printf("2: Proc %d one send time: %lf\n", proc_rank, te - ts);
    MPI_Barrier(MPI_COMM_WORLD);
 
    // Tell MPI to shut down.
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}