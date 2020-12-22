
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "msg_utils.h"
#include "CrossNodeData.h"

void allocParaCND(CrossNodeData *data, int node_num, int delay) 
{
	assert(delay > 0);
	data->_node_num = node_num;
	data->_delay = delay;

	int size = delay * node_num;
	int num_p_1 = node_num + 1;

	data->_recv_offset = (int *)malloc(sizeof(int) * (num_p_1));
	data->_recv_num = (int *)malloc(sizeof(int) * size);
	data->_recv_data = NULL;

	data->_send_offset = (int *)malloc(sizeof(int) * (num_p_1));
	data->_send_num = (int *)malloc(sizeof(int) * size);
	data->_send_data = NULL;

	resetCND(data);
}

void resetCND(CrossNodeData *data)
{
	int size = data->_delay * data->_node_num;
	memset(data->_recv_num, 0, sizeof(int) * size);
	memset(data->_send_num, 0, sizeof(int) * size);
}

void allocDataCND(CrossNodeData *data)
{
	int num = data->_node_num;
	int data_size = data->_recv_offset[num];
	if (data_size > 0) {
		data->_recv_data = (int*)malloc(sizeof(int)*(data_size));
		memset(data->_recv_data, 0, sizeof(int) * (data_size));
	}

	data_size = data->_send_offset[num];
	if (data_size > 0) {
		data->_send_data = (int*)malloc(sizeof(int)*(data_size));
		memset(data->_send_data, 0, sizeof(int) * (data_size));
	}
}

void freeCND(CrossNodeData *data)
{
	free(data->_recv_offset);
	free(data->_recv_num);
	free(data->_recv_data);

	free(data->_send_offset);
	free(data->_send_num);
	free(data->_send_data);

	data->_node_num = 0;
	data->_delay = 0;
	free(data);
	data = NULL;
}

int sendCND(CrossNodeData *cnd, int dst, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(&(cnd->_node_num), 1, MPI_INT, dst, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(cnd->_delay), 1, MPI_INT, dst, tag+1, comm);
	assert(ret == MPI_SUCCESS);

	// int size = cnd->_delay * cnd->_node_num;
	int num_p_1 = cnd->_node_num + 1;
	ret = MPI_Send(cnd->_recv_offset, num_p_1, MPI_INT, dst, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(cnd->_send_offset, num_p_1, MPI_INT, dst, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	return ret;
}

CrossNodeData * recvCND(int src, int tag, MPI_Comm comm)
{
	CrossNodeData *cnd = (CrossNodeData *)malloc(sizeof(CrossNodeData));
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(cnd->_node_num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(cnd->_delay), 1, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);

	int size = cnd->_delay * cnd->_node_num;
	int num_p_1 = cnd->_node_num + 1;
	cnd->_recv_offset = (int *)malloc(sizeof(int)*num_p_1);
	ret = MPI_Recv(cnd->_recv_offset, num_p_1, MPI_INT, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);
	cnd->_send_offset = (int *)malloc(sizeof(int)*num_p_1);
	ret = MPI_Recv(cnd->_send_offset, num_p_1, MPI_INT, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);

	cnd->_recv_num = (int *)malloc(sizeof(int)*(size));
	cnd->_send_num = (int *)malloc(sizeof(int)*(size));
	resetCND(cnd);

	allocDataCND(cnd);

	return cnd;
}

int generateCND(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, int *send_data, int *send_offset, int *send_num, int node_num, int time, int gFiredTableCap, int min_delay, int delay)
{
	int delay_idx = time % (conn->maxDelay+1);
	int fired_size = firedTableSizes[delay_idx];
	for (int node=0; node<node_num; node++) {
		for (int idx=0; idx<fired_size; idx++) {
			int nid = firedTable[gFiredTableCap * delay_idx + idx];
			int tmp = idx2index[nid];
			if (tmp >= 0) {
				int map_nid = crossnode_index2idx[tmp*node_num+node];
				if (map_nid >= 0) {
					send_data[send_offset[node] + send_num[node*min_delay+delay]]= map_nid;
					send_num[node*min_delay+delay]++;
				}
			}
		}
	}
	return 0;
}

#define ASYNC

int msg_cnd(CrossNodeData *cnd, int *send_num, int *recv_num, MPI_Request *request)
{

	int node_num = cnd->_node_num;
	int delay = cnd->_delay;
	for (int i=0; i<node_num; i++) {
		send_num[i] = cnd->_send_num[i*delay+delay-1];
	}

	int num_size = delay * node_num;

	print_mpi_x32(cnd->_send_num, num_size, "Send Num");

	MPI_Alltoall(cnd->_send_num, num_size, MPI_INT, cnd->_recv_num, num_size, MPI_INT, MPI_COMM_WORLD);

	print_mpi_x32(cnd->_recv_num, num_size, "Recv Num");

	for (int i=0; i<node_num; i++) {
		recv_num[i] = cnd->_recv_num[i*delay+delay-1];
	}

#ifdef ASYNC
	int ret = MPI_Ialltoallv(cnd->_send_data, send_num, cnd->_send_offset , MPI_INT, cnd->_recv_data, recv_num, cnd->_recv_offset, MPI_INT, MPI_COMM_WORLD, request);
	assert(ret == MPI_SUCCESS);
#else
	int ret = MPI_Alltoallv(cnd->_send_data, c_send_num, cnd->_send_offset, MPI_INT, cnd->_recv_data, recv_num, cnd->_recv_offset, MPI_INT, MPI_COMM_WORLD);
	assert(ret == MPI_SUCCESS);
#endif

	return ret;
}
