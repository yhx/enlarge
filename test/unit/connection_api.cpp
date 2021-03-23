/* This program is writen by qp09.
 * usually just for fun.
 * Tue December 15 2015
 */

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "../../include/BSim.h"
#include "../../src/net/GNetwork.h"

using std::vector;
using ::testing::AtLeast;
using ::testing::ElementsAreArray;


TEST(ConnectionAPITest, SynapseTemplTest) {

	real dt = 1.0e-4;
	Network c0(dt), c1(dt);

	Population *pn0 = c0.createPopulation(2, LIFNeuron(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, dt));
	Population *pn1 = c0.createPopulation(3, LIFNeuron(2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, dt));
	Population *pn2 = c0.createPopulation(1, LIFNeuron(3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, dt));

	Population *pn0_1 = c1.createPopulation(2, LIFNeuron(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, dt));
	Population *pn1_1 = c1.createPopulation(3, LIFNeuron(2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, dt));
	Population *pn2_1 = c1.createPopulation(1, LIFNeuron(3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, dt));

	real weight0[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5};
	real weight1[] = {2.0, 2.1, 2.2};
	real delay0[] = {1e-4, 2e-4, 1e-4, 2e-4, 1e-4, 3e-4};
	real delay1[] = {1e-4, 2e-4, 3e-4};

	StaticSynapse s0 = StaticSynapse(weight0, delay0, 0.0, dt, 6);
	StaticSynapse s1 = StaticSynapse(weight1, delay1, 0.0, dt, 6);

	//Network.connect(population1, population2, weight_array, delay_array, Exec or Inhi array, num)
	c0.connect(pn0, pn1, weight0, delay0, NULL, 6);
	c0.connect(pn1, pn2, weight1, delay1, NULL, 3);

	c1.connect(pn0_1, pn1_1, s0);
	c1.connect(pn1_1, pn2_1, s1);


	// SGSim sg(&c, 1.0e-4);

	DistriNetwork *n0 = c0.buildNetworks(SimInfo(1.0e-4));
	DistriNetwork *n1 = c1.buildNetworks(SimInfo(1.0e-4));

	ASSERT_TRUE(compareGNetwork(n0->_network, n1->_network));

}

int main(int argc, char **argv)
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
} 
