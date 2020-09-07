
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
	CrossNodeData *net = (CrossNodeData *)malloc(sizeof(CrossNodeData));
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->_node_num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	net->_recv_offset = (int *)malloc(sizeof(int)*(net->_node_num+1));
	ret = MPI_Recv(net->_recv_offset, net->_node_num+1, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	net->_send_offset = (int *)malloc(sizeof(int)*(net->_node_num+1));
	ret = MPI_Recv(net->_send_offset, net->_node_num+1, MPI_INT, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	net->_recv_num = (int *)malloc(sizeof(int)*(net->_node_num));
	net->_send_num = (int *)malloc(sizeof(int)*(net->_node_num));
	resetCND(net);

	allocDataCND(net);

	return net;
}

int generateCND(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, int *send_data, int *send_offset, int *send_num, int node_num, int time, int gFiredTableCap)
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
					send_data[send_offset[node] + send_num[node]]= map_nid;
					send_num[node]++;
				}
			}
		}
	}
	return 0;
}
