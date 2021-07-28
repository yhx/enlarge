
#include <stdio.h>

#include "../../msg_utils/msg_utils/CrossMap.h"
#include "../../msg_utils/msg_utils/CrossSpike.h"
#include "../msg_utils/CrossNodeMap.h"
#include "../msg_utils/CrossNodeData.h"
#include "convert.h"

CrossMap * convert2crossmap(CrossNodeMap * cnm)
{
	size_t num = cnm->_num;
	size_t size = cnm->_crossSize;
	CrossMap *cm = new CrossMap(num, size);

	for (size_t i=0; i<num; i++) {
		cm->_idx2index[i] = cnm->_idx2index[i];
	}

	for (size_t i=0; i<size; i++) {
		cm->_index2ridx[i] = cnm->_crossnodeIndex2idx[i];
	}

	return cm;
}

CrossSpike * convert2crossspike(CrossNodeData *cnd, int proc_rank, int gpu_num)
{
	int proc_num = cnd->_node_num;
	int delay = cnd->_min_delay;
	CrossSpike *cs = NULL;
	
	if (gpu_num > 0) {
		cs = new CrossSpike(proc_rank, proc_num, delay, gpu_num);
	} else {
		cs = new CrossSpike(proc_rank, proc_num, delay);
	}

	for (int i=0;  i<proc_num + 1; i++) {
		cs->_recv_offset[i] = cnd->_recv_offset[i];
		cs->_send_offset[i] = cnd->_send_offset[i];
	}

	for (int i=0;  i<proc_num * (delay + 1); i++) {
		cs->_recv_start[i] = cnd->_recv_start[i];
		cs->_send_start[i] = cnd->_send_start[i];
	}

	for (int i=0;  i<proc_num; i++) {
		cs->_recv_num[i] = cnd->_recv_num[i];
		cs->_send_num[i] = cnd->_send_num[i];
	}

	cs->alloc();

	for (int i=0;  i<cs->_recv_offset[proc_num]; i++) {
		cs->_recv_data[i] = cnd->_recv_data[i];
	}

	for (int i=0;  i<cs->_send_offset[proc_num]; i++) {
		cs->_send_data[i] = cnd->_send_data[i];
	}

	return cs;
}
