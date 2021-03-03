/* This header file is writen by qp09
 * usually just for fun
 * Thu October 22 2015
 */
#ifndef NEURON_H
#define NEURON_H

#include "../interface/Model.h"

class Neuron : public Model {
public:
	Neuron(Type type, size_t num, size_t offset=0) : Model(type, num, offset) {
	}

	virtual ~Neuron() = 0;

	virtual int append(const Neuron *n, size_t num) = 0;
};

#endif /* NEURON_H */

