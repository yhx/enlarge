
#ifndef CROSSNODEDATA_H
#define CROSSNODEDATA_H

#include <string>
#include "mpi.h"

#include "../base/constant.h"
#include "../net/Connection.h"

using std::string;

// Assuming node number is N, then the offset and num parameter both have N elements. offset[i] means the offset location on data array for ith node, num[i] records the actual data recived from/sended to the ith node. offset[N] is the size of data array.
struct CrossNodeData {
	int _node_num;
	int _min_delay;

	// int _recv_size; _recv_offset[_node_num];
	// cap _node_num + 1
	integer_t *_recv_offset;
	// cap _node_num * (delay+1)
	integer_t *_recv_start;
	// cap _node_num
    integer_t *_recv_num;
	uinteger_t *_recv_data;

	// int send_size;
	// cap _node_num + 1
	integer_t *_send_offset;
	// cap _node_num * (delay+1)
	integer_t *_send_start;
	// cap _node_num * delay
	integer_t *_send_num;
	uinteger_t *_send_data;
};

void allocParaCND(CrossNodeData *data, int node_num, int delay);
void allocDataCND(CrossNodeData *data);
void resetCND(CrossNodeData *data);
void freeCND(CrossNodeData *data);

bool isEqualCND(CrossNodeData *data1, CrossNodeData *data2);

int sendCND(CrossNodeData *data, int dst, int tag, MPI_Comm comm);
CrossNodeData * recvCND(int src, int tag, MPI_Comm comm);

int saveCND(CrossNodeData *data, const string &path);
CrossNodeData * loadCND(const string &path);

CrossNodeData * copyCNDtoGPU(CrossNodeData * data);
int freeCNDGPU(CrossNodeData *data);

int generateCND(integer_t *idx2index, integer_t *crossnode_index2idx, CrossNodeData *cnd, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t gFiredTableCap, int max_delay, int min_delay, int node_num, int time);

int msg_cnd(CrossNodeData *cnd, MPI_Request *request);

int update_cnd(CrossNodeData *cnd, int curr_delay, MPI_Request *request);

int log_cnd(CrossNodeData *cnd, int time, FILE *sfile, FILE *rfile);

void cudaGenerateCND(integer_t *idx2index, integer_t *crossnode_index2idx, CrossNodeData *cnd, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, int max_delay, int min_delay, int node_num, int time, int gridSize, int blockSize); 

int update_cnd_gpu(CrossNodeData *gpu, CrossNodeData *cpu, int curr_delay, MPI_Request *request);

int reset_cnd_gpu(CrossNodeData *gpu, CrossNodeData *cpu);

#endif // CROSSNODEDATA_H
