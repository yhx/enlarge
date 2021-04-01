/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "../../include/BSim.h"
#include "../../src/msg_utils/msg_utils.h"

using namespace std;

using std::vector;
using ::testing::AtLeast;
using ::testing::ElementsAreArray;

const int NODE_NUM = 2;
const int N = 2;
const int DELAY = 3;

int node_id = -1;
int node_num = -1;

DistriNetwork *network = NULL;
CrossNodeData *data = NULL;

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
	ASSERT_EQ(network->_crossnodeMap->_num, 6);
	ASSERT_EQ(network->_crossnodeMap->_crossSize, NODE_NUM*2);

	if (node_id == 0) {
		ASSERT_THAT(vector<int>(network->_crossnodeMap->_idx2index, network->_crossnodeMap->_idx2index+network->_crossnodeMap->_num),
				ElementsAreArray({-1, -1, 0, 1, -1, -1}));
		ASSERT_THAT(vector<int>(network->_crossnodeMap->_crossnodeIndex2idx, network->_crossnodeMap->_crossnodeIndex2idx+network->_crossnodeMap->_crossSize),
				ElementsAreArray({-1, 4, -1, 5}));
	} else {
		ASSERT_THAT(vector<int>(network->_crossnodeMap->_idx2index, network->_crossnodeMap->_idx2index+network->_crossnodeMap->_num),
				ElementsAreArray({-1, -1, 0, 1, -1, -1}));
		ASSERT_THAT(vector<int>(network->_crossnodeMap->_crossnodeIndex2idx, network->_crossnodeMap->_crossnodeIndex2idx+network->_crossnodeMap->_crossSize),
				ElementsAreArray({4, -1, 5, -1}));
	}
}

TEST(MPITEST, CNDTest) {
	ASSERT_EQ(data->_node_num, NODE_NUM);
	ASSERT_EQ(data->_min_delay, DELAY);

	if (node_id == 0) {
		ASSERT_THAT(vector<int>(data->_recv_offset, data->_recv_offset+NODE_NUM+1),
				ElementsAreArray({0, 0, 6}));
		ASSERT_THAT(vector<int>(data->_send_offset, data->_send_offset+NODE_NUM+1),
				ElementsAreArray({0, 0, 6}));
	} else {
		ASSERT_THAT(vector<int>(data->_recv_offset, data->_recv_offset+NODE_NUM+1),
				ElementsAreArray({0, 6, 6}));
		ASSERT_THAT(vector<int>(data->_send_offset, data->_send_offset+NODE_NUM+1),
				ElementsAreArray({0, 6, 6}));
	}
}

TEST(MPITEST, MSGTest) {
	if (node_id == 0) {
		ASSERT_THAT(vector<int>(data->_send_start, data->_send_start+NODE_NUM*(DELAY+1)),
				ElementsAreArray({0, 0, 0, 0, 0, 2, 4, 5}));
		ASSERT_THAT(vector<int>(data->_send_num, data->_send_num+NODE_NUM),
				ElementsAreArray({0, 5}));
		ASSERT_THAT(vector<int>(data->_send_data, data->_send_data+data->_send_num[1]),
				ElementsAreArray({4, 5, 4, 5, 5}));

		ASSERT_THAT(vector<int>(data->_recv_start, data->_recv_start+NODE_NUM*(DELAY+1)),
				ElementsAreArray({0, 0, 0, 0, 0, 2, 4, 5}));
		ASSERT_THAT(vector<int>(data->_recv_num, data->_recv_num+NODE_NUM),
				ElementsAreArray({0, 5}));
		ASSERT_THAT(vector<int>(data->_recv_data, data->_recv_data+data->_recv_num[1]),
				ElementsAreArray({4, 5, 4, 5, 4}));
	} else {
		ASSERT_THAT(vector<int>(data->_send_start, data->_send_start+NODE_NUM*(DELAY+1)),
				ElementsAreArray({0, 2, 4, 5, 0, 0, 0, 0}));
		ASSERT_THAT(vector<int>(data->_send_num, data->_send_num+NODE_NUM),
				ElementsAreArray({5, 0}));
		ASSERT_THAT(vector<int>(data->_send_data, data->_send_data+data->_send_num[0]),
				ElementsAreArray({4, 5, 4, 5, 4}));

		ASSERT_THAT(vector<int>(data->_recv_start, data->_recv_start+NODE_NUM*(DELAY+1)),
				ElementsAreArray({0, 2, 4, 5, 0, 0, 0, 0}));
		ASSERT_THAT(vector<int>(data->_recv_num, data->_recv_num+NODE_NUM),
				ElementsAreArray({5, 0}));
		ASSERT_THAT(vector<int>(data->_recv_data, data->_recv_data+data->_recv_num[0]),
				ElementsAreArray({4, 5, 4, 5, 5}));
	}
}

TEST(MPITEST, SAVE_LOAD_TEST) {
	MNSim mn("multi_node_test");
	ASSERT_TRUE(compareDistriNet(mn._network_data, network));
	ASSERT_TRUE(isEqualCND(mn._data, data));
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	real dt = 1e-4;
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
	data = sg._data;


	uinteger_t fire_tbl[12] = {0, 1, 2, 3, 2, 3, 0, 0, 3, 1, 0, 0};
	uinteger_t fire_tbl_size[3] = {4, 2, 3};

	uinteger_t fire_tbl1[12] = {2, 3, 0, 0, 0, 1, 2, 3, 2, 0, 0, 0};
	uinteger_t fire_tbl_size1[3] = {2, 4, 1};

	for (int time = 0; time<DELAY; time++) {
		int max_delay = network->_network->ppConnections[0]->maxDelay;
		if (node_id == 0) {
			generateCND(network->_crossnodeMap->_idx2index, network->_crossnodeMap->_crossnodeIndex2idx, data, fire_tbl, fire_tbl_size, N*2, max_delay, data->_min_delay, NODE_NUM, time);
		} else {
			generateCND(network->_crossnodeMap->_idx2index, network->_crossnodeMap->_crossnodeIndex2idx, data, fire_tbl1, fire_tbl_size1, N*2, max_delay, data->_min_delay, NODE_NUM, time);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Request request_t;
		int curr_delay = time % data->_min_delay;
		update_cnd(data, curr_delay, &request_t);

		MPI_Barrier(MPI_COMM_WORLD);
	}

	sg.save_net("multi_node_test");
	MPI_Barrier(MPI_COMM_WORLD);

	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
} 
