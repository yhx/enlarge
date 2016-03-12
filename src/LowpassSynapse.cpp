/* This program is writen by qp09.
 * usually just for fun.
 * Fri October 23 2015
 */

#include <math.h>

#include "utils/json/json.h"
#include "LowpassSynapse.h"

const Type LowpassSynapse::type = Lowpass;

LowpassSynapse::LowpassSynapse(ID id, real weight, real delay = 0, real tau_syn = 0)
{
	this->weight = weight;
	this->delay = delay;
	this->tau_syn = tau_syn;
	this->id = id;
	this->monitored = false;
	file = NULL;
}

LowpassSynapse::~LowpassSynapse()
{
	if (file != NULL) {
		fflush(file);
		fclose(file);
	}

	delay_step.clear();
}

void LowpassSynapse::setDst(NeuronBase *p) {
	pDest = p;
}

int LowpassSynapse::reset(SimInfo &info) {
	I_syn = 0;
	init(info.dt);

	return 0;
}

int LowpassSynapse::init(real dt) {
	_dt = dt;
	if (tau_syn > 0.008) {
		C1 = -0.90483742;
		_C1 = 0.09516258;
	} else {
		C1 = -0.81873075;
		_C1 = 0.18126925;
	}

	return 0;
}

int LowpassSynapse::update(SimInfo &info)
{
	
	list<int>::iterator iter;

	while (!delay_step.empty()) /* && (delay_step.front() <= 0)) */ {
		real i_tmp = weight/_dt;
		I_syn = -C1 * I_syn;
		I_syn += weight/_dt*_C1;
		if (monitored) {
			if (file != NULL) {
				fprintf(file, "Cycle %d: weighted %f, lowpass %f\n", info.currCycle, i_tmp, this->I_syn); 
			}
		}
		pDest->recv(I_syn);
		delay_step.pop_front();
	}

	//for (iter = delay_step.begin(); iter != delay_step.end(); iter++) {
	//	*iter = *iter - 1;
	//}

	return 0;
}

int LowpassSynapse::recv()
{
	//printf("Syn: %d_%d\n", this->getID().groupId, this->getID().id);
	delay_step.push_back((int)(delay/_dt));
	if (monitored) {
		if (file != NULL) {
			fprintf(file, "recived %d\n", (int)(delay/_dt)); 
		}
	}

	return 0;
}

ID LowpassSynapse::getID()
{
	return id;
}

Type LowpassSynapse::getType()
{
	return type;
}

void LowpassSynapse::monitorOn() 
{
	monitored = true;
}

void LowpassSynapse::monitor(SimInfo &info)
{
	if (monitored) {
		if (file == NULL) {
			char filename[128];
			sprintf(filename, "Synapse_%d.log", id.id);
			file = fopen(filename, "w+");
			if (file == NULL) {
				printf("Open file %s failed\n", filename);
				return;
			}
			fprintf(file, "W: %f, D: %f, T:%f\n", weight, delay, tau_syn);
			fprintf(file, "C1: %f, ", C1);
			fprintf(file, "dt: %f\n", _dt);
		}
	}

	return;
}

size_t LowpassSynapse::getSize()
{
	//return sizeof(GLowpassSynapses);
	return 0;
}

int LowpassSynapse::getData(void *data)
{
	Json::Value *p = (Json::Value *)data;
	(*p)["weight"] = weight;
	(*p)["delay"] = delay;

	return 0;
}

int LowpassSynapse::hardCopy(void *data, int idx)
{
	//GLowpassSynapses *p = (GLowpassSynapses *)data;
	//p->pID[idx] = id;
	//p->pType[idx] = type;
	//p->p_weight[idx] = weight;
	//p->p_delay[idx] = delay;
	//p->p_C1[idx] = C1;
	//p->p__C1[idx] = _C1;
	//p->p_tau_syn[idx] = tau_syn;
	//p->p_I_syn[idx] = I_syn;
	//p->p__dt[idx] = _dt;
	return 1;
}
