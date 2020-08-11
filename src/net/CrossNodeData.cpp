
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "CrossNodeData.h"

void allocParaCND(CrossNodeData *data, int node_num) 
{
	data->_node_num = node_num;

	data->_recv_offset = (int *)malloc(sizeof(int) * (node_num + 1));
	data->_recv_num = (int *)malloc(sizeof(int) * node_num);
	data->_recv_data = NULL;

	data->_send_offset = (int *)malloc(sizeof(int) * (node_num + 1));
	data->_send_num = (int *)malloc(sizeof(int) * node_num);
	data->_send_data = NULL;

	resetCND(data);
}

void resetCND(CrossNodeData *data)
{
	memset(data->_recv_num, 0, sizeof(int) * (data->_node_num));
	memset(data->_send_num, 0, sizeof(int) * (data->_node_num));
}

void allocDataCND(CrossNodeData *data)
{
	if (data->_recv_offset[data->_node_num] > 0) {
		data->_recv_data = (int*)malloc(sizeof(int)*(data->_recv_offset[data->_node_num]));
		memset(data->_recv_data, 0, sizeof(int) * (data->_recv_offset[data->_node_num]));
	}

	if (data->_send_offset[data->_node_num] > 0) {
		data->_send_data = (int*)malloc(sizeof(int)*(data->_send_offset[data->_node_num]));
		memset(data->_send_data, 0, sizeof(int) * (data->_send_offset[data->_node_num]));
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
	free(data);
	data = NULL;
}

int sendCND(CrossNodeData *data, int dst, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(&(data->_node_num), 1, MPI_INT, dst, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->_recv_offset, data->_node_num+1, MPI_INT, dst, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->_send_offset, data->_node_num+1, MPI_INT, dst, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	return ret;
}

CrossNodeData * recvCND(int src, int tag, MPI_Comm comm)
{
	CrossNodeData *ret = (CrossNodeData *)malloc(sizeof(CrossNodeData));
	MPI_Status status;
	MPI_Recv(&(ret->_node_num), 1, MPI_INT, src, tag, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	ret->_recv_offset = (int *)malloc(sizeof(int)*(ret->_node_num+1));
	MPI_Recv(ret->_recv_offset, ret->_node_num+1, MPI_INT, src, tag+1, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	ret->_send_offset = (int *)malloc(sizeof(int)*(ret->_node_num+1));
	MPI_Recv(ret->_send_offset, ret->_node_num+1, MPI_INT, src, tag+2, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	ret->_recv_num = (int *)malloc(sizeof(int)*(ret->_node_num));
	ret->_send_num = (int *)malloc(sizeof(int)*(ret->_node_num));
	resetCND(ret);

	allocDataCND(ret);

	return ret;
}
