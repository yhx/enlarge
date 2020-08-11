
#ifndef CROSSNODEDATA_H
#define CROSSNODEDATA_H

#include "mpi.h"

// Assuming node number is N, then the offset and num parameter both have N elements. offset[i] means the offset location on data array for ith node, num[i] records the actual data recived from/sended to the ith node. offset[N] is the size of data array.
struct CrossNodeData {
	int _node_num;

	// int _recv_size; _recv_offset[_node_num];
	int *_recv_offset;
	int *_recv_num;
	int *_recv_data;

	// int send_size;
	int *_send_offset;
	int *_send_num;
	int *_send_data;
};

void allocParaCND(CrossNodeData *data, int node_num);
void allocDataCND(CrossNodeData *data);
void resetCND(CrossNodeData *data);
void freeCND(CrossNodeData *data);


int sendCND(CrossNodeData *data, int dst, int tag, MPI_Comm comm);
CrossNodeData * recvCND(int src, int tag, MPI_Comm comm);

CrossNodeData * copyCNDtoGPU(CrossNodeData * data);
int freeCNDGPU(CrossNodeData *data);

#endif // CROSSNODEDATA_H
