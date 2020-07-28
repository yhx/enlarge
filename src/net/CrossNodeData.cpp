
#include "CrossNodeData.h"

void allocParaCND(CrossNodeData *data, int node_num) {
	data->_node_num = node_num;

	data->_recv_offset = (int *)malloc(sizeof(int) * (node_num + 1));
	data->_recv_num = (int *)malloc(sizeof(int) * node_num);
	data->_recv_data = NULL;

	data->_send_offset = (int *)malloc(sizeof(int) * (node_num + 1));
	data->_send_num = (int *)malloc(sizeof(int) * node_num);
	data->_send_data = NULL;

	resetCND(data);
}

void resetCND(CrossNodeData *data) {
	memset(data->_recv_num, 0, sizeof(int) * node_num);
	memset(data->_send_num, 0, sizeof(int) * node_num);
}
