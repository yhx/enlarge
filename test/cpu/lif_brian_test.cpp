/* This program is writen by qp09.
 * usually just for fun.
 * Tue October 13 2015
 */

#include "../../include/BSim.h"

int main(int argc, char **argv)
{
	real dt = 1.0e-3;
	LIFNeuron n = LIFNeuron(0.0, 0.0, 0.0, 1.0e-1, 10.0e-3, 0.0, 1.0, 1.0, 15.0e-3, 2.0e-3, dt);
	Network c(dt);
	c.createPopulation(0, 1, n);
	SingleThreadSimulator s(&c, dt);
	s.run(0.1);

	return 0;
} 



