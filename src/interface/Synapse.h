/* This header file is writen by qp09
 * usually just for fun
 * Mon September 28 2015
 */

#ifndef SYNAPSE_H
#define SYNAPSE_H

#include <math.h>

#include "../interface/Neuron.h"

class Synapse: public Model {
public:
	Synapse(Type type, int num, int offset=0) : Model(type, num, offset) {}
	virtual ~Synapse() = 0;

	const vector<unsigned int> & delay() {
		return _delay;
	}

	// inline real getRealDelay() {
	// 	return _delay;
	// }

	// inline int getDelaySteps(real dt) {
	// 	return static_cast<int>(round(_delay/dt));
	// }
	
	virtual int append(const Synapse *s, size_t num=0) = 0;

protected:
	// vector<real> _weight;
	vector<unsigned int> _delay;
};

// class Greater {
// 	bool operator()(Synapse *a, Synapse *b) const {
// 		return (a->getRealDelay()) > (b->getRealDelay());
// 	}
// };

#endif /* SYNAPSE_H */

