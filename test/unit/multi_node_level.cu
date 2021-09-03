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

const int NODE_NUM = 2;
const int N = 2;
const int DELAY = 3;

const real dt = 1e-4;

int node_id = -1;
int node_num = -1;

DistriNetwork *network = NULL;
CrossMap *cm = NULL;
CrossSpike *msg = NULL;

TEST(MPITest, InfoTest) {
	ASSERT_EQ(node_num, NODE_NUM);
}

TEST(MPITEST, NetTest) {
	ASSERT_EQ(network->_simCycle, 100);
	ASSERT_EQ(network->_nodeIdx, node_id);
	ASSERT_EQ(network->_nodeNum, node_num);
	ASSERT_FLOAT_EQ(network->_dt, 1e-4);
}

TEST(MPITEST, CNMTest) {
	ASSERT_EQ(cm->_num, 6);
	ASSERT_EQ(cm->_cross_size, NODE_NUM*2);

	if (node_id == 0) {
		ASSERT_THAT(vector<int>(cm->_idx2index, cm->_idx2index+cm->_num),
				EQUALS({-1, -1, 0, 1, -1, -1}));
		ASSERT_THAT(vector<int>(cm->_index2ridx, cm->_index2ridx+cm->_cross_size),
				EQUALS({-1, 4, -1, 5}));
	} else {
		ASSERT_THAT(vector<int>(cm->_idx2index, cm->_idx2index+cm->_num),
				EQUALS({-1, -1, 0, 1, -1, -1}));
		ASSERT_THAT(vector<int>(cm->_index2ridx, cm->_index2ridx+cm->_cross_size),
				EQUALS({4, -1, 5, -1}));
	}
}

TEST(MPITEST, CNDTest) {
	ASSERT_EQ(msg->_proc_num, NODE_NUM);
	ASSERT_EQ(msg->_min_delay, DELAY);

	if (node_id == 0) {
		ASSERT_THAT(vector<int>(msg->_recv_offset, msg->_recv_offset+NODE_NUM+1),
				EQUALS({0, 0, 6}));
		ASSERT_THAT(vector<int>(msg->_send_offset, msg->_send_offset+NODE_NUM+1),
				EQUALS({0, 0, 6}));
	} else {
		ASSERT_THAT(vector<int>(msg->_recv_offset, msg->_recv_offset+NODE_NUM+1),
				EQUALS({0, 6, 6}));
		ASSERT_THAT(vector<int>(msg->_send_offset, msg->_send_offset+NODE_NUM+1),
				EQUALS({0, 6, 6}));
	}
}

TEST(MPITEST, MSGTest) {
	if (node_id == 0) {
		ASSERT_THAT(vector<int>(msg->_send_start, msg->_send_start+NODE_NUM*(DELAY+1)),
				EQUALS({0, 0, 0, 0, 0, 2, 4, 5}));
		ASSERT_THAT(vector<int>(msg->_send_num, msg->_send_num+NODE_NUM),
				EQUALS({0, 5}));
		ASSERT_THAT(vector<int>(msg->_send_data, msg->_send_data+msg->_send_num[1]),
				EQUALS({4, 5, 4, 5, 5}));

		ASSERT_THAT(vector<int>(msg->_recv_start, msg->_recv_start+NODE_NUM*(DELAY+1)),
				EQUALS({0, 0, 0, 0, 0, 2, 4, 5}));
		ASSERT_THAT(vector<int>(msg->_recv_num, msg->_recv_num+NODE_NUM),
				EQUALS({0, 5}));
		ASSERT_THAT(vector<int>(msg->_recv_data, msg->_recv_data+msg->_recv_num[1]),
				EQUALS({4, 5, 4, 5, 4}));
	} else {
		ASSERT_THAT(vector<int>(msg->_send_start, msg->_send_start+NODE_NUM*(DELAY+1)),
				EQUALS({0, 2, 4, 5, 0, 0, 0, 0}));
		ASSERT_THAT(vector<int>(msg->_send_num, msg->_send_num+NODE_NUM),
				EQUALS({5, 0}));
		ASSERT_THAT(vector<int>(msg->_send_data, msg->_send_data+msg->_send_num[0]),
				EQUALS({4, 5, 4, 5, 4}));

		ASSERT_THAT(vector<int>(msg->_recv_start, msg->_recv_start+NODE_NUM*(DELAY+1)),
				EQUALS({0, 2, 4, 5, 0, 0, 0, 0}));
		ASSERT_THAT(vector<int>(msg->_recv_num, msg->_recv_num+NODE_NUM),
				EQUALS({5, 0}));
		ASSERT_THAT(vector<int>(msg->_recv_data, msg->_recv_data+msg->_recv_num[0]),
				EQUALS({4, 5, 4, 5, 5}));
	}
}

TEST(MPITEST, SAVE_LOAD_TEST) {
	MNSim mn("multi_node_test", dt);
	ASSERT_TRUE(compareDistriNet(mn._network_data, network));
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	Network c(dt);

	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);


	if (node_id == 0) {
		Population *pn0 = c.createPopulation(N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 100.0e-1, dt));
		Population *pn1 = c.createPopulation(N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));
		Population *pn2 = c.createPopulation(N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));
		Population *pn3 = c.createPopulation(N, LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3, dt));

		real * weight0 = NULL;
		real * weight1 = NULL;
		real * delay = NULL;
		real * delay1 = NULL;

		printf("GENERATE DATA...\n");
		weight0 = getConstArray((real)1e-5, N*N);
		weight1 = getConstArray((real)2e-5, N*N);
		delay = getConstArray((real)(DELAY*1e-4), N*N);
		delay1 = getConstArray((real)(DELAY*1e-4), N*N);
		printf("GENERATE DATA FINISHED\n");

		//Network.connect(population1, population2, weight_array, delay_array, Exec or Inhi array, num)
		c.connect(pn0, pn1, weight0, delay, NULL, N*N);
		c.connect(pn1, pn2, weight1, delay, NULL, N*N);
		c.connect(pn2, pn3, weight1, delay, NULL, N*N);
		c.connect(pn3, pn0, weight1, delay1, NULL, N*N);
	}

	MNSim sg(&c, 1.0e-4);

	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &node_num);
	
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, node_id, node_num);

	SimInfo info(1.0e-4);
	info.save_mem = true;
	int sim_cycle = 100;

	to_attach();
	sg.distribute(info, sim_cycle);
	network = sg._network_data;
	CrossNodeData * data = sg._data;

	cm = convert2crossmap(network->_crossnodeMap);
	msg = convert2crossspike(data, node_id, 2);

	cm->to_gpu();
	msg->to_gpu();


	uinteger_t fire_tbl[12] = {0, 1, 2, 3, 2, 3, 0, 0, 3, 1, 0, 0};
	uinteger_t fire_tbl_size[3] = {4, 2, 3};

	uinteger_t fire_tbl1[12] = {2, 3, 0, 0, 0, 1, 2, 3, 2, 0, 0, 0};
	uinteger_t fire_tbl_size1[3] = {2, 4, 1};

	uinteger_t *fire_tbl_g = TOGPU(fire_tbl, 12);
	uinteger_t *fire_tbl_size_g = TOGPU(fire_tbl_size, 3);

	uinteger_t *fire_tbl1_g = TOGPU(fire_tbl1, 12);
	uinteger_t *fire_tbl_size1_g = TOGPU(fire_tbl_size1, 3);


	for (int time = 0; time<DELAY; time++) {
		int max_delay = network->_network->ppConnections[0]->maxDelay;
		if (node_id == 0) {
			msg->fetch_gpu(cm, fire_tbl_g, fire_tbl_size_g, N*2, node_num, max_delay, time, 1, 128);
		} else {
			msg->fetch_gpu(cm, fire_tbl1_g, fire_tbl_size1_g, N*2, node_num, max_delay, time, 1, 128);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		msg->update_gpu(time);

		MPI_Barrier(MPI_COMM_WORLD);
	}

	sg.save_net("multi_node_test");
	MPI_Barrier(MPI_COMM_WORLD);

	// copyFromGPU(msg->_recv_start, msg->_gpu_array->_recv_start, msg->_min_delay * msg->_proc_num + msg->_proc_num);
	msg->from_gpu();

	int ret = CATCH_RET();
	MPI_Finalize();
	return ret;
} 
