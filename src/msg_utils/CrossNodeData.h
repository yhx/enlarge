
#ifndef CROSSNODEDATA_H
#define CROSSNODEDATA_H

#include "mpi.h"

#include "../net/Connection.h"

// Assuming node number is N, then the offset and num parameter both have N elements. offset[i] means the offset location on data array for ith node, num[i] records the actual data recived from/sended to the ith node. offset[N] is the size of data array.
struct CrossNodeData {
	int _node_num;
	int _min_delay;

	// int _recv_size; _recv_offset[_node_num];
	// cap _node_num + 1
	int *_recv_offset;
	// cap _node_num * (delay+1)
	int *_recv_start;
	// cap _node_num
	int *_recv_num;
	int *_recv_data;

	// int send_size;
	// cap _node_num + 1
	int *_send_offset;
	// cap _node_num * (delay+1)
	int *_send_start;
	// cap _node_num * delay
	int *_send_num;
	int *_send_data;
};

void allocParaCND(CrossNodeData *data, int node_num, int delay);
void allocDataCND(CrossNodeData *data);
void resetCND(CrossNodeData *data);
void freeCND(CrossNodeData *data);


int sendCND(CrossNodeData *data, int dst, int tag, MPI_Comm comm);
CrossNodeData * recvCND(int src, int tag, MPI_Comm comm);

int saveCND(CrossNodeData *data, FILE *f);
CrossNodeData * loadCND(FILE *f);

CrossNodeData * copyCNDtoGPU(CrossNodeData * data);
int freeCNDGPU(CrossNodeData *data);

int generateCND(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, CrossNodeData *cnd, int node_num, int time, int gFiredTableCap, int min_delay, int delay);

int msg_cnd(CrossNodeData *cnd, MPI_Request *request);

int update_cnd(CrossNodeData *cnd, int curr_delay, MPI_Request *request);

int log_cnd(CrossNodeData *cnd, int time, FILE *sfile, FILE *rfile);

void cudaGenerateCND(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, CrossNodeData *cnd, int node_num, int time, int delay, int gridSize, int blockSize);

int update_cnd_gpu(CrossNodeData *gpu, CrossNodeData *cpu, int curr_delay, MPI_Request *request);

int reset_cnd_gpu(CrossNodeData *gpu, CrossNodeData *cpu);

#endif // CROSSNODEDATA_H
