/* This header file is writen by qp09
 * usually just for fun
 * Thu October 22 2015
 */
#ifndef NEURON_H
#define NEURON_H

#include "../interface/Model.h"

class Neuron : public Model {
public:
	Neuron(Type type, int num, int offset=0) : Model(num, type, offset) {
	}

	virtual ~Neuron() = 0;
};

#endif /* NEURON_H */

