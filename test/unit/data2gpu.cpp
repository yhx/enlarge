/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "../catch2/catch.h"

#include "../../include/BSim.h"

using std::vector;

GNetwork *net = NULL;
GNetwork *net2 = NULL;

TEST(NetworkTest, NeuronTest) {
	ASSERT_EQ(net->nTypeNum, 1);
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
	// ASSERT_THAT(
	// 		vector<real>(n->pV_e, n->pV_e + net->pNeuronNums[net->nTypeNum]), 
	// 		APPROX({0, 0, 0, 0, 0, 0})
	// 		);
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
	// ASSERT_THAT(
	// 		vector<real>(n->pV_i, n->pV_i + net->pNeuronNums[net->nTypeNum]), 
	// 		APPROX({0, 0, 0, 0, 0, 0})
	// 		);
}


TEST(NetworkTest, SynapseTest) {
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
	// ASSERT_THAT(
	// 		vector<int>(s->pDst, s->pDst + 9), 
	// 		EQUALS({2, 4, 3, 3, 2, 4, 5, 5, 5})
	// 		);
}
 
TEST(NetworkTest, ConnectionTest) {
	Connection *c = net->ppConnections[0];

	ASSERT_EQ(c->nNum, 6);
	ASSERT_EQ(c->sNum, 9);
	ASSERT_EQ(c->maxDelay, 3);
	ASSERT_EQ(c->minDelay, 1);

	ASSERT_THAT(
			vector<int>(c->pDelayNum, c->pDelayNum + c->nNum * (c->maxDelay-c->minDelay+1)), 
			// EQUALS({2, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0})
			EQUALS({2, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0})
			);
	ASSERT_THAT(
			vector<int>(c->pDelayStart, c->pDelayStart + c->nNum * (c->maxDelay-c->minDelay+1)), 
			// EQUALS({0, 2, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9})
			EQUALS({0, 2, 3, 4, 4, 4, 4, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9})
			);

	ASSERT_THAT(
			vector<int>(c->pSidMap, c->pSidMap + c->sNum), 
			EQUALS({0, 2, 4, 6, 1, 3, 7, 5, 8})
			);

	ASSERT_THAT(
			vector<int>(c->dst, c->dst + c->sNum), 
			EQUALS({2, 4, 3, 5, 3, 2, 5, 4, 5})
			// EQUALS({2, 4, 3, 3, 2, 4, 5, 5, 5})
			);
}

TEST(NetworkTest, NetTest) {
	ASSERT_TRUE(compareGNetwork(net, net2));
}


int main(int argc, char **argv)
{
	real dt = 1.0e-4;
	Network c(dt);
	Population *pn0 = c.createPopulation(2, LIFNeuron(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, dt));
	Population *pn1 = c.createPopulation(3, LIFNeuron(2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, dt));
	Population *pn2 = c.createPopulation(1, LIFNeuron(3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, dt));

	real weight0[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5};
	real weight1[] = {2.0, 2.1, 2.2};
	real delay0[] = {1e-4, 2e-4, 1e-4, 2e-4, 1e-4, 3e-4};
	real delay1[] = {1e-4, 2e-4, 3e-4};

	//Network.connect(population1, population2, weight_array, delay_array, Exec or Inhi array, num)
	c.connect(pn0, pn1, weight0, delay0, NULL, 6);
	c.connect(pn1, pn2, weight1, delay1, NULL, 3);

	//SGSim sg(&c, 1.0e-4);

	net = c.buildNetwork(SimInfo(dt));
	GNetwork * netGPU = copyGNetworkToGPU(net);

	net2 = deepcopyGNetwork(net);

	fetchGNetworkFromGPU(net2, netGPU);

	int ret = Catch::Session().run(argc, argv);
	return ret;
} 
