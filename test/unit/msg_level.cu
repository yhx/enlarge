
/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "../catch2/catch.h"

#include "../../include/BSim.h"
#include "../../src/msg_utils/convert.h"
#include "../../msg_utils/msg_utils/msg_utils.h"
#include "../../msg_utils/msg_utils/CrossMap.h"
#include "../../msg_utils/msg_utils/CrossSpike.h"
#include "../../msg_utils/helper/helper_gpu.h"

using std::vector;

const int N = 2;
const int CAP = 20;
const int DELAY = 1;
// const int NODE_NUM = 2;

// const real dt = 1e-4;

int proc_rank = 0;

CrossMap *cm = NULL;
CrossSpike *cs = NULL;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int proc_num = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

	to_attach();

	int tmp[CAP] = {-1, -1, -1, -1, -1, 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, 5, 6, 7, 8, 9};
	int tmp2[CAP] = {-1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1, 16, -1, 17, -1, 18, -1, 19};

	nid_t table[40] = {0};
	nid_t table_sizes[2] = {0};

	switch (proc_rank) {
		case 0:
			cm = new CrossMap(CAP, CAP);
			for (int i=0; i<CAP; i++) {
				cm->_idx2index[i] = tmp[i];
				cm->_index2ridx[i] = tmp2[i];
			}
			table[0] = 19;
			table_sizes[0] = 1;
			break;
		case 1:
			cm = new CrossMap(CAP, 0);
			for (int i=0; i<CAP; i++) {
				cm->_idx2index[i] = -1;
			}
			break;
		default:
			return proc_rank;
	}


	cs = new CrossSpike(proc_rank, proc_num, DELAY, 1);
	cs->_recv_offset[0] = 0;
	cs->_send_offset[0] = 0;

	for (int i=0; i<proc_num; i++) {
		cs->_recv_offset[i+1] = cs->_recv_offset[i] + DELAY * (N-1);
		cs->_send_offset[i+1] = cs->_send_offset[i] + DELAY * (N-1);
	}

	cs->alloc();

	char name[1024];
	sprintf(name, "%s_%d", argv[0], proc_rank);

	cm->log(name);
	cm->to_gpu();

	cs->to_gpu();


	nid_t *table_gpu = TOGPU(table,  (DELAY+1) * CAP);
	nid_t *table_sizes_gpu = TOGPU(table_sizes, DELAY+1);

	for (int t=0; t<DELAY; t++) {
		cs->fetch_gpu(cm, (nid_t *)table_gpu, (nsize_t *)table_sizes_gpu, CAP, proc_num, DELAY, t, 1, 32);
		cs->update_gpu(t);
		cs->log_gpu(t, name); 
		cs->upload_gpu((nid_t *)table_gpu, (nsize_t *)table_sizes_gpu, (nsize_t *)table_sizes, CAP, DELAY, t, 1, 32);
	}


	COPYFROMGPU(table, table_gpu, (DELAY+1) * CAP);
	COPYFROMGPU(table_sizes, table_sizes_gpu, DELAY+1);

	MPI_Barrier(MPI_COMM_WORLD);

	for (int i=0; i<DELAY+1; i++) {
		printf("Rank %d:%d :", proc_rank, table_sizes[i]);
		for (int j=0; j<table_sizes[i]; j++) {
			printf("%d ", table[j + i * CAP]);
		}
		printf("\n");
	}


	// int result = Catch::Session().run( argc, argv );
	int result = 0;

	MPI_Finalize();

	return result;
}
