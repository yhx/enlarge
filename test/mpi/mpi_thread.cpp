#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pthread.h>

int proc_num;
int proc_rank;
int thread_num = 2;
int thread_datasize = 10;

pthread_barrier_t barrier;

struct Para {
    int thread_id;
};

void *mpi_prog(void *tmp) {
    Para *para = (Para *)tmp;
    int thread_id = para->thread_id;
    int *send = (int *)malloc(sizeof(int) * thread_datasize);
    for (int i = 0; i < thread_datasize; ++i) {
        send[i] = i + proc_num * proc_rank * thread_datasize + thread_id * thread_datasize;
    }
    int *recv = (int *)malloc(sizeof(int) * thread_datasize * thread_num * proc_num);
    MPI_Alltoall(send, thread_datasize, MPI_INT, recv, thread_datasize, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    pthread_barrier_wait(&barrier);

    for (int i = 0; i < proc_num; ++i) {
        if (proc_rank == i) {
            for (int j = 0; j < thread_num; ++j) {
                if (j == thread_id) {
                    printf("proc: %d thread: %d : ", proc_rank, thread_id);
                    for (int i = 0; i < thread_datasize * thread_num * proc_num; ++i) {
                        printf("%d ", recv[i]);
                    }
                }
                pthread_barrier_wait(&barrier);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    return 0;
}

/**
 * @brief Illustrates how to initialise the MPI environment with multithreading
 * support and ask for the MPI_THREAD_MULTIPLE level.
 * @details This application initialised MPI and asks for the 
 * MPI_THREAD_MULTIPLE thread support level. It then compares it with the
 * thread support level provided by the MPI implementation.
 **/
int main(int argc, char* argv[])
{
    // Initilialise MPI and ask for thread support
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);

    if(provided < MPI_THREAD_MULTIPLE)
    {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    else
    {
        printf("The threading support level corresponds to that demanded.\n");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    

    pthread_t* thread_ids = (pthread_t *)malloc(thread_num * sizeof(pthread_t));
    Para *paras = (Para *)malloc(thread_num * sizeof(Para));
    pthread_barrier_init(&barrier, NULL, thread_num);

    for (int i = 0; i < thread_num; ++i) {
        paras[i].thread_id = i;
        pthread_create(&(thread_ids[i]), NULL, &mpi_prog, (void *)&(paras[i]));
    }

    for (int i = 0; i < thread_num; ++i) {
        pthread_join(thread_ids[i], NULL);
    }
    pthread_barrier_destroy(&barrier);
 
    // Tell MPI to shut down.
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}