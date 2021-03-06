/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "../catch2/catch.h"

#include "../../include/BSim.h"
#include "../../msg_utils/helper/helper_c.h"
#include "../../src/net/GNetwork.h"

using std::vector;

GNetwork *net = NULL;
Network * network = NULL;

// #define EQUALS(val) Equals(vector<int>((val)))
// #define EQUALT(val) Equals(vector<Type>((val)))
// #define APPROX(val) Approx(vector<real>((val)))


TEST_CASE("NeuronTest", "") {
	REQUIRE(net->nTypeNum == 1);

	ASSERT_THAT(
			vector<Type>(net->pNTypes, net->pNTypes + net->nTypeNum), 
			EQUALT({LIF})
			);
	ASSERT_THAT(
			vector<int>(net->pNeuronNums, net->pNeuronNums + net->nTypeNum + 1), 
			EQUALS({0, 6})
			);

	LIFData *n = (LIFData*)net->ppNeurons[0];

	ASSERT_THAT(
			vector<int>(n->pRefracTime, n->pRefracTime + net->pNeuronNums[net->nTypeNum]), 
			EQUALS({15000, 15000, 25000, 25000, 25000, 35000})
			);
	ASSERT_THAT(
			vector<int>(n->pRefracStep, n->pRefracStep + net->pNeuronNums[net->nTypeNum]), 
			EQUALS({0, 0, 0, 0, 0, 0})
			);

	ASSERT_THAT(
			vector<real>(n->pV_thresh, n->pV_thresh + net->pNeuronNums[net->nTypeNum]), 
			APPROX({1.8, 1.8, 2.8, 2.8, 2.8, 3.8})
			);
	ASSERT_THAT(
			vector<real>(n->pI_i, n->pI_i + net->pNeuronNums[net->nTypeNum]), 
			APPROX({0, 0, 0, 0, 0, 0})
			);
	ASSERT_THAT(
			vector<real>(n->pV_tmp, n->pV_tmp + net->pNeuronNums[net->nTypeNum]), 
			APPROX({0.00022465541388783095, 0.00022465541388783095, 0.00021357148557399341, 0.00021357148557399341, 0.00021357148557399341, 0.00020916842316864859})
			);

	ASSERT_THAT(
			vector<real>(n->pC_i, n->pC_i + net->pNeuronNums[net->nTypeNum]), 
			APPROX({7.6749377e-05, 7.6749377e-05, 4.3661708e-05, 4.3661708e-05, 4.3661708e-05, 3.0296025e-05})
			);
	ASSERT_THAT(
			vector<real>(n->pCe, n->pCe + net->pNeuronNums[net->nTypeNum]), 
			APPROX({0.99993747, 0.99993747, 0.99996156, 0.99996156, 0.99996156, 0.99997222})
			);
	ASSERT_THAT(
			vector<real>(n->pC_m, n->pC_m + net->pNeuronNums[net->nTypeNum]), 
			APPROX({0.99992859, 0.99992859, 0.99995834, 0.99995834, 0.99995834, 0.99997061})
			);
	ASSERT_THAT(
			vector<real>(n->pV_m, n->pV_m + net->pNeuronNums[net->nTypeNum]), 
			APPROX({1.0, 1.0, 2.0, 2.0, 2.0, 3.0})
			);
	ASSERT_THAT(
			vector<real>(n->pV_reset, n->pV_reset + net->pNeuronNums[net->nTypeNum]), 
			APPROX({1.2, 1.2, 2.2, 2.2, 2.2, 3.2})
			);
	ASSERT_THAT(
			vector<real>(n->pCi, n->pCi + net->pNeuronNums[net->nTypeNum]), 
			APPROX({0.99994117, 0.99994117, 0.99996299, 0.99996299, 0.99996299, 0.999973})
			);
	ASSERT_THAT(
			vector<real>(n->pI_e, n->pI_e + net->pNeuronNums[net->nTypeNum]), 
			APPROX({0, 0, 0, 0, 0, 0})
			);
	ASSERT_THAT(
			vector<real>(n->pC_e, n->pC_e + net->pNeuronNums[net->nTypeNum]), 
			APPROX({7.6514014e-05, 7.6514014e-05, 4.3661741e-05, 4.3661741e-05, 4.3661741e-05, 2.9845702e-05})
			);
}

TEST_CASE("SynapseTest", "") {
	ASSERT_EQ(net->sTypeNum, 1);
	ASSERT_THAT(
			vector<Type>(net->pSTypes, net->pSTypes + net->sTypeNum), 
			EQUALT({Static})
			);
	ASSERT_THAT(
			vector<int>(net->pSynapseNums, net->pSynapseNums + net->sTypeNum + 1), 
			EQUALS({0, 9})
			);

	StaticData *s = (StaticData*)net->ppSynapses[0];
	ASSERT_THAT(
			vector<real>(s->pWeight, s->pWeight + 9), 
			APPROX({1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.1, 2.2})
			);
}

TEST_CASE("ConnectionTest", "") {
	Connection *c = net->ppConnections[0];

	ASSERT_EQ(c->nNum, 6);
	ASSERT_EQ(c->sNum, 9);
	ASSERT_EQ(c->maxDelay, 3);
	ASSERT_EQ(c->minDelay, 1);

	ASSERT_THAT(
			vector<int>(c->pDelayStart, c->pDelayStart + c->nNum * (c->maxDelay-c->minDelay+1)), 
			// EQUALS({0, 2, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9})
			EQUALS({0, 2, 3, 4, 4, 4, 4, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9})
			);

	ASSERT_THAT(
			vector<int>(c->pDelayNum, c->pDelayNum + c->nNum * (c->maxDelay-c->minDelay+1)), 
			// EQUALS({2, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0})
			EQUALS({2, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0})
			);

	ASSERT_THAT(
			vector<int>(c->pSidMap, c->pSidMap + c->sNum), 
			EQUALS({0, 2, 4, 6, 1, 3, 7, 5, 8})
			);

	ASSERT_THAT(
			vector<int>(c->dst, c->dst + c->sNum), 
			EQUALS({2, 4, 3, 11, 3, 2, 11, 4, 11})
			);
}

TEST_CASE("BuildTest1", "") {
	SimInfo info(1e-4);
	GNetwork * n1 = network->buildNetwork(info);
	DistriNetwork* n2 = network->buildNetworks(info);
	ASSERT_TRUE(compareGNetwork(n1, n2->_network));
}

TEST_CASE("BuildTest2") {
	SimInfo info(1e-4);
	network->set_node_num(2);
	DistriNetwork* n = network->buildNetworks(info, RoundRobin);
	ASSERT_EQ(n[0]._simCycle, 0);
	ASSERT_EQ(n[0]._nodeIdx, 0);
	ASSERT_EQ(n[0]._nodeNum, 2);
	ASSERT_FLOAT_EQ(n[0]._dt, 1e-4);
	ASSERT_EQ(n[1]._simCycle, 0);
	ASSERT_EQ(n[1]._nodeIdx, 1);
	ASSERT_EQ(n[1]._nodeNum, 2);
	ASSERT_FLOAT_EQ(n[1]._dt, 1e-4);

	GNetwork *n0 = n[0]._network;
	ASSERT_EQ(n0->nTypeNum, 1);
	ASSERT_THAT(
			vector<Type>(n0->pNTypes, n0->pNTypes + n0->nTypeNum), 
			EQUALT({LIF})
			);
	ASSERT_THAT(
			vector<int>(n0->pNeuronNums, n0->pNeuronNums + n0->nTypeNum + 1), 
			EQUALS({0, 3})
			);
	ASSERT_EQ(n0->sTypeNum, 1);
	ASSERT_THAT(
			vector<Type>(n0->pSTypes, n0->pSTypes + n0->sTypeNum), 
			EQUALT({Static})
			);
	ASSERT_THAT(
			vector<int>(n0->pSynapseNums, n0->pSynapseNums + n0->sTypeNum + 1), 
			EQUALS({0, 4})
			);


	GNetwork *n1 = n[1]._network;
	ASSERT_EQ(n1->nTypeNum, 1);
	ASSERT_THAT(
			vector<Type>(n1->pNTypes, n1->pNTypes + n1->nTypeNum), 
			EQUALT({LIF})
			);
	ASSERT_THAT(
			vector<int>(n1->pNeuronNums, n1->pNeuronNums + n1->nTypeNum + 1), 
			EQUALS({0, 3})
			);
	ASSERT_EQ(n1->sTypeNum, 1);
	ASSERT_THAT(
			vector<Type>(n1->pSTypes, n1->pSTypes + n1->sTypeNum), 
			EQUALT({Static})
			);
	ASSERT_THAT(
			vector<int>(n1->pSynapseNums, n1->pSynapseNums + n1->sTypeNum + 1), 
			EQUALS({0, 5})
			);


	LIFData *n0_ = (LIFData*)n0->ppNeurons[0];
	ASSERT_THAT(
			vector<int>(n0_->pRefracTime, n0_->pRefracTime + n0->pNeuronNums[n0->nTypeNum]), 
			EQUALS({15000, 25000, 25000})
			);
	ASSERT_THAT(
			vector<int>(n0_->pRefracStep, n0_->pRefracStep + n0->pNeuronNums[n0->nTypeNum]), 
			EQUALS({0, 0, 0})
			);
	ASSERT_THAT(
			vector<real>(n0_->pV_thresh, n0_->pV_thresh + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({1.8, 2.8, 2.8})
			);
	ASSERT_THAT(
			vector<real>(n0_->pI_i, n0_->pI_i + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({0, 0, 0})
			);
	ASSERT_THAT(
			vector<real>(n0_->pV_tmp, n0_->pV_tmp + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({0.00022465541388783095, 0.00021357148557399341, 0.00021357148557399341})
			);
	// ASSERT_THAT(
	// 		vector<real>(n0_->pV_e, n0_->pV_e + n0->pNeuronNums[n0->nTypeNum]), 
	// 		APPROX({0, 0, 0, 0, 0})
	// 		);
	ASSERT_THAT(
			vector<real>(n0_->pC_i, n0_->pC_i + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({7.6749377e-05,  4.3661708e-05, 4.3661708e-05})
			);
	ASSERT_THAT(
			vector<real>(n0_->pCe, n0_->pCe + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({0.99993747, 0.99996156, 0.99996156})
			);
	ASSERT_THAT(
			vector<real>(n0_->pC_m, n0_->pC_m + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({0.99992859, 0.99995834, 0.99995834})
			);
	ASSERT_THAT(
			vector<real>(n0_->pV_m, n0_->pV_m + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({1.0, 2.0, 2.0})
			);
	ASSERT_THAT(
			vector<real>(n0_->pV_reset, n0_->pV_reset + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({1.2, 2.2, 2.2})
			);
	ASSERT_THAT(
			vector<real>(n0_->pCi, n0_->pCi + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({0.99994117, 0.99996299, 0.99996299})
			);
	ASSERT_THAT(
			vector<real>(n0_->pI_e, n0_->pI_e + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({0, 0, 0})
			);
	ASSERT_THAT(
			vector<real>(n0_->pC_e, n0_->pC_e + n0->pNeuronNums[n0->nTypeNum]), 
			APPROX({7.6514014e-05, 4.3661741e-05, 4.3661741e-05})
			);
	// ASSERT_THAT(
	// 		vector<real>(n0_->pV_i, n0_->pV_i + n0->pNeuronNums[n0->nTypeNum]), 
	// 		APPROX({0, 0, 0, 0, 0})
	// 		);

	LIFData *n1_ = (LIFData*)n1->ppNeurons[0];

	ASSERT_THAT(
			vector<int>(n1_->pRefracTime, n1_->pRefracTime + n1->pNeuronNums[n1->nTypeNum]), 
			EQUALS({15000, 25000, 35000})
			);
	ASSERT_THAT(
			vector<int>(n1_->pRefracStep, n1_->pRefracStep + n1->pNeuronNums[n1->nTypeNum]), 
			EQUALS({0, 0, 0})
			);
	ASSERT_THAT(
			vector<real>(n1_->pV_thresh, n1_->pV_thresh + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({1.8, 2.8, 3.8})
			);
	ASSERT_THAT(
			vector<real>(n1_->pI_i, n1_->pI_i + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({0, 0, 0})
			);
	ASSERT_THAT(
			vector<real>(n1_->pV_tmp, n1_->pV_tmp + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({0.00022465541388783095, 0.00021357148557399341, 0.00020916842316864859})
			);
	// ASSERT_THAT(
	// 		vector<real>(n1_->pV_e, n1_->pV_e + n1->pNeuronNums[n1->nTypeNum]), 
	// 		APPROX({0})
	// 		);
	ASSERT_THAT(
			vector<real>(n1_->pC_i, n1_->pC_i + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({7.6749377e-05, 4.3661708e-05, 3.0296025e-05})
			);
	ASSERT_THAT(
			vector<real>(n1_->pCe, n1_->pCe + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({0.99993747, 0.99996156, 0.99997222})
			);
	ASSERT_THAT(
			vector<real>(n1_->pC_m, n1_->pC_m + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({0.99992859, 0.99995834, 0.99997061})
			);
	ASSERT_THAT(
			vector<real>(n1_->pV_m, n1_->pV_m + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({1.0, 2.0, 3.0})
			);
	ASSERT_THAT(
			vector<real>(n1_->pV_reset, n1_->pV_reset + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({1.2, 2.2, 3.2})
			);
	ASSERT_THAT(
			vector<real>(n1_->pCi, n1_->pCi + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({0.99994117, 0.99996299, 0.999973})
			);
	ASSERT_THAT(
			vector<real>(n1_->pI_e, n1_->pI_e + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({0, 0, 0})
			);
	ASSERT_THAT(
			vector<real>(n1_->pC_e, n1_->pC_e + n1->pNeuronNums[n1->nTypeNum]), 
			APPROX({7.6514014e-05, 4.3661741e-05, 2.9845702e-05})
			);
	// ASSERT_THAT(
	// 		vector<real>(n1_->pV_i, n1_->pV_i + n1->pNeuronNums[n1->nTypeNum]), 
	// 		APPROX({0})
	// 		);

	StaticData *s0 = (StaticData*)n0->ppSynapses[0];
	ASSERT_THAT(
			vector<real>(s0->pWeight, s0->pWeight + n0->pSynapseNums[n0->sTypeNum]), 
			APPROX({1.0, 1.2, 1.3, 1.5})
			);
	// ASSERT_THAT(
	// 		vector<int>(s0->pDst, s0->pDst + n0->pSynapseNums[n0->sTypeNum]), 
	// 		EQUALS({2, 4, 3, 3, 2, 4})
	// 		);

	StaticData *s1 = (StaticData*)n1->ppSynapses[0];
	ASSERT_THAT(
			vector<real>(s1->pWeight, s1->pWeight + n1->pSynapseNums[n1->sTypeNum]), 
			APPROX({1.4, 2.0, 2.1, 1.1, 2.2})
			);

	CrossNodeMap * c0_ = n[0]._crossnodeMap;
	ASSERT_EQ(c0_->_crossSize, n[0]._nodeNum * 3);
	ASSERT_THAT(
			vector<int>(c0_->_idx2index, c0_->_idx2index + c0_->_num), 
			// vector<int>(c0_->_idx2index, c0_->_idx2index + 5), 
			EQUALS({0, 1, 2, -1})
			);
	ASSERT_THAT(
			vector<int>(c0_->_crossnodeIndex2idx, c0_->_crossnodeIndex2idx + c0_->_crossSize), 
			EQUALS({-1, 3, -1, 4, -1, 5})
			);

	CrossNodeMap * c1_ = n[1]._crossnodeMap;
	ASSERT_EQ(c1_->_crossSize, n[1]._nodeNum * 1);
	ASSERT_THAT(
			vector<int>(c1_->_idx2index, c1_->_idx2index + c1_->_num), 
			EQUALS({0, -1, -1, -1, -1, -1})
			);

	ASSERT_THAT(
			vector<int>(c1_->_crossnodeIndex2idx, c1_->_crossnodeIndex2idx + c1_->_crossSize), 
			EQUALS({3, -1})
			);
	// ASSERT_EQ(c1_->_crossnodeIndex2idx, nullptr);

	Connection *c0 = n0->ppConnections[0];
	ASSERT_EQ(c0->nNum, 4);
	ASSERT_EQ(c0->sNum, 4);
	ASSERT_EQ(c0->maxDelay, 3);
	ASSERT_EQ(c0->minDelay, 1);
	ASSERT_THAT(
			vector<int>(c0->pDelayNum, c0->pDelayNum + c0->nNum * (c0->maxDelay-c0->minDelay+1)), 
			EQUALS({2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1})
			);
	ASSERT_THAT(
			vector<int>(c0->pDelayStart, c0->pDelayStart + c0->nNum * (c0->maxDelay-c0->minDelay+1)+1), 
			EQUALS({0, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4})
			);

	ASSERT_THAT(
			vector<int>(c0->pSidMap, c0->pSidMap + c0->sNum), 
			EQUALS({0, 1, 2, 3})
			);

	ASSERT_THAT(
			vector<int>(c0->dst, c0->dst + c0->sNum), 
			EQUALS({1, 2, 1, 2})
			);

	Connection *c1 = n1->ppConnections[0];
	REQUIRE(c1->nNum == 6);
	REQUIRE(c1->sNum == 5);
	REQUIRE(c1->maxDelay == 3);
	REQUIRE(c1->minDelay == 1);
	ASSERT_THAT(
			vector<int>(c1->pDelayNum, c1->pDelayNum + c1->nNum * (c1->maxDelay-c1->minDelay+1)), 
			EQUALS({1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1})
			);
	ASSERT_THAT(
			vector<int>(c1->pDelayStart, c1->pDelayStart + c1->nNum * (c1->maxDelay-c1->minDelay+1)+1), 
			EQUALS({0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5})
			);

	ASSERT_THAT(
			vector<int>(c1->pSidMap, c1->pSidMap + c1->sNum), 
			EQUALS({0, 1, 2, 3, 4})
			);

	ASSERT_THAT(
			vector<int>(c1->dst, c1->dst + c1->sNum), 
			EQUALS({1, 5, 5, 1, 5})
			);
}

TEST_CASE("SaveLoadTest", "") {
	system("mkdir tmp");
	saveGNetwork(net, "./tmp");
	GNetwork *t = loadGNetwork("./tmp");
	ASSERT_TRUE(compareGNetwork(net, t));
}


int main(int argc, char **argv)
{
	real dt = 1.0e-4;
	Network c(dt);
	network = &c;
	Population *pn0 = c.createPopulation(2, LIFNeuron(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, dt));
	Population *pn1 = c.createPopulation(3, LIFNeuron(2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, dt));
	Population *pn2 = c.createPopulation(1, LIFNeuron(3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, dt));

	real weight0[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5};
	real weight1[] = {2.0, 2.1, 2.2};
	real delay0[] = {1e-4, 2e-4, 1e-4, 2e-4, 1e-4, 3e-4};
	real delay1[] = {1e-4, 2e-4, 3e-4};

	SpikeType type[] = {Inh, Inh, Inh};

	//Network.connect(population1, population2, weight_array, delay_array, Exec or Inhi array, num)
	c.connect(pn0, pn1, weight0, delay0, NULL, 6);
	c.connect(pn1, pn2, weight1, delay1, type, 3);

	// SGSim sg(&c, 1.0e-4);

	net = network->buildNetwork(SimInfo(1.0e-4));

	int ret = Catch::Session().run(argc, argv);
	return ret;
} 
