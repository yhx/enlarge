/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "../../include/BSim.h"

using namespace std;

using std::vector;
using ::testing::AtLeast;
using ::testing::ElementsAreArray;

const int NODE_NUM = 2;
const int N = 2;
const int DELAY = 2;

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
	ASSERT_EQ(data->_delay, DELAY);

	if (node_id == 0) {
		ASSERT_THAT(vector<int>(data->_recv_offset, data->_recv_offset+NODE_NUM+1),
				ElementsAreArray({0, 0, 4}));
		ASSERT_THAT(vector<int>(data->_send_offset, data->_send_offset+NODE_NUM+1),
				ElementsAreArray({0, 0, 4}));
	} else {
		ASSERT_THAT(vector<int>(data->_recv_offset, data->_recv_offset+NODE_NUM+1),
				ElementsAreArray({0, 4, 4}));
		ASSERT_THAT(vector<int>(data->_send_offset, data->_send_offset+NODE_NUM+1),
				ElementsAreArray({0, 4, 4}));
	}
}

TEST(MPITEST, MSGTest) {
	if (node_id == 0) {
		ASSERT_THAT(vector<int>(data->_send_num, data->_send_num+NODE_NUM*DELAY),
				ElementsAreArray({0, 0, 2, 4}));
		ASSERT_THAT(vector<int>(data->_send_data, data->_send_data+NODE_NUM*DELAY),
				ElementsAreArray({4, 5, 4, 5}));

		ASSERT_THAT(vector<int>(data->_recv_num, data->_recv_num+NODE_NUM*DELAY),
				ElementsAreArray({0, 0, 2, 4}));
		ASSERT_THAT(vector<int>(data->_recv_data, data->_recv_data+NODE_NUM*DELAY),
				ElementsAreArray({4, 5, 4, 5}));
	} else {
		ASSERT_THAT(vector<int>(data->_send_num, data->_send_num+NODE_NUM*DELAY),
				ElementsAreArray({2, 4, 0, 0}));
		ASSERT_THAT(vector<int>(data->_send_data, data->_send_data+NODE_NUM*DELAY),
				ElementsAreArray({4, 5, 4, 5}));

		ASSERT_THAT(vector<int>(data->_recv_num, data->_recv_num+NODE_NUM*DELAY),
				ElementsAreArray({2, 4, 0, 0}));
		ASSERT_THAT(vector<int>(data->_recv_data, data->_recv_data+NODE_NUM*DELAY),
				ElementsAreArray({4, 5, 4, 5}));
	}
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	Network c;

	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);


	if (node_id == 0) {
		Population *pn0 = c.createPopulation(N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 100.0e-1), 1.0, 1.0));
		Population *pn1 = c.createPopulation(N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3), 1.0, 1.0));
		Population *pn2 = c.createPopulation(N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3), 1.0, 1.0));
		Population *pn3 = c.createPopulation(N, LIF_curr_exp(LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 50.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 0.0e-3), 1.0, 1.0));

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

	sg.distribute(&network, &data, info, sim_cycle);

	int *c_send_num = (int *)malloc(sizeof(int)*NODE_NUM);
	int *c_recv_num = (int *)malloc(sizeof(int)*NODE_NUM);

	int fire_tbl[8] = {0, 1, 2, 3, 2, 3, 0, 0};
	int fire_tbl_size[2] = {4, 2};

	int fire_tbl1[8] = {2, 3, 0, 0, 0, 1, 2, 3};
	int fire_tbl_size1[2] = {2, 4};

	for (int time = 0; time<2; time++) {
		int curr_delay = time % data->_delay;
		if (node_id == 0) {
			generateCND(network->_network->pConnection, fire_tbl, fire_tbl_size, network->_crossnodeMap->_idx2index, network->_crossnodeMap->_crossnodeIndex2idx, data->_send_data, data->_send_offset, data->_send_num, NODE_NUM, time, N*2, data->_delay, curr_delay);
		} else {
			generateCND(network->_network->pConnection, fire_tbl1, fire_tbl_size1, network->_crossnodeMap->_idx2index, network->_crossnodeMap->_crossnodeIndex2idx, data->_send_data, data->_send_offset, data->_send_num, NODE_NUM, time, N*2, data->_delay, curr_delay);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Request request_t;
		if (curr_delay >= DELAY-1) {
			msg_cnd(data, c_send_num, c_recv_num, &request_t);
		} else {
			for (int i=0; i<NODE_NUM; i++) {
				data->_send_num[i*DELAY+curr_delay+1] = data->_send_num[i*DELAY+curr_delay];
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
} 
