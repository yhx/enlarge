
#include <assert.h>

#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "Network.h"
// #include "../neuron/array/ArrayNeuron.h"
// #include "../neuron/array/GArrayNeurons.h"

// TODO uncomment to support ArrayNeuron
// void arrangeFireArray(vector<int> &fire_array, vector<int> &start_loc, PopulationBase *popu)
// {
// 	size_t num = popu->getNum();
// 	for (size_t i=0; i<num; i++) {
// 		ArrayNeuron *p = dynamic_cast<ArrayNeuron*>(popu->getNeuron(i));
// 		vector<int> &vec = p->getFireTimes();
// 		start_loc.push_back(fire_array.size());
// 		fire_array.insert(fire_array.end(), vec.begin(), vec.end());
// 	}
// }

// TODO uncomment to support ArrayNeuron
// void arrangeArrayNeuron(vector<int> &fire_array, vector<int> &start_loc, GArrayNeurons *p, int num)
// {
// 	assert(num == (int)start_loc.size());
// 	for (int i=0; i<num; i++) {
// 		p->p_start[i] = start_loc[i];
// 		p->p_end[i] += p->p_start[i];
// 		if (i > 0) {
// 			assert(p->p_end[i-1] == p->p_start[i]);
// 		}
// 	}
// 	assert(p->p_end[num-1] == (int)fire_array.size());
// 	p->p_fire_time = static_cast<int*>(malloc(sizeof(int) * fire_array.size()));
// 	std::copy(fire_array.begin(), fire_array.end(), p->p_fire_time);
// }

GNetwork* Network::buildNetwork(SimInfo &info)
{
	vector<Population *>::iterator piter;
	vector<Neuron *>::iterator niter;
	vector<Synapse *>::iterator siter;
	// vector<int> array_neuron_start;
	// vector<int> array_neuron_fire_times;

	int neuronTypeNum = nTypes.size();
	int synapseTypeNum = sTypes.size();
	int maxDelaySteps = static_cast<int>(round(maxDelay/info.dt));
	int minDelaySteps = static_cast<int>(round(minDelay/info.dt));
	int deltaDelay = maxDelaySteps - minDelaySteps + 1;

	GNetwork * ret = allocNetwork(neuronTypeNum, synapseTypeNum);

	for (int i=0; i<neuronTypeNum; i++) {
		ret->pNTypes[i] = nTypes[i];

		ret->ppNeurons[i] = allocType[nTypes[i]](neuronNums[i]);
		assert(ret->ppNeurons[i] != NULL);

		int idx = 0;
		for (piter = pPopulations.begin(); piter != pPopulations.end();  piter++) {
			Population * p = *piter;
			if (p->getType() == nTypes[i]) {
				size_t copied = p->hardCopy(ret->ppNeurons[i], idx, ret->pNeuronNums[i], info);
				idx += copied;

				// TODO uncomment to support array
				// if (p->getType() == Array) {
				// 	arrangeFireArray(array_neuron_fire_times, array_neuron_start, p);
				// }

			}
		}

		assert(idx == neuronNums[i]);

		// TODO uncomment to support array
		// if (nTypes[i] == Array) {
		// 	arrangeArrayNeuron(array_neuron_fire_times, array_neuron_start, static_cast<GArrayNeurons*>(pN), idx);
		// }

		ret->pNeuronNums[i+1] = idx + ret->pNeuronNums[i];
	}
	assert(ret->pNeuronNums[ret->nTypeNum] == totalNeuronNum);

	for (int i=0; i<synapseTypeNum; i++) {
		ret->pSTypes[i] = sTypes[i];

		ret->ppSynapses[i] = allocType[sTypes[i]](synapseNums[i]);
		assert(ret->ppSynapses[i] != NULL);

		int idx = 0;
		for (auto piter = pPopulations.begin(); piter != pPopulations.end(); piter++) {
			Population * p = *piter;
			for (int nidx=0; nidx<p->getNum(); nidx++) {
				const vector<Synapse *> &s_vec = p->locate(nidx)->getSynapses();
				for (int delay_t=0; delay_t < deltaDelay; delay_t++) {
					for (auto siter = s_vec.begin(); siter != s_vec.end(); siter++) {
						if ((*siter)->getDelaySteps(info.dt) == delay_t + minDelaySteps) {
							if ((*siter)->getType() == sTypes[i]) {
								//int sid = (*iter)->getID();
								//assert(synapseIdx < totalSynapseNum);
								//synapseIdx++;
								int copied = (*siter)->hardCopy(ret->ppSynapses[i], idx, ret->pSynapseNums[i], info);
								idx += copied;
							}
						}
					}
				}
			}
		}
		//for (siter = pSynapses.begin(); siter != pSynapses.end();  siter++) {
		//	SynapseBase * p = *siter;
		//	if (p->getType() == sTypes[i]) {
		//		int copied = p->hardCopy(pS, idx, pSynapsesNum[i]);
		//		idx += copied;
		//	}
		//}

		assert(idx == synapseNums[i]);
		ret->pSynapseNums[i+1] = idx + ret->pSynapseNums[i];
	}
	assert(ret->pSynapseNums[ret->sTypeNum] == totalSynapseNum);

	logMap();

	ret->pConnection = allocConnection(totalNeuronNum, totalSynapseNum, maxDelaySteps, minDelaySteps);

	int synapseIdx = 0;
	for (auto piter = pPopulations.begin(); piter != pPopulations.end(); piter++) {
		Population * p = *piter;
		for (int i=0; i<p->getNum(); i++) {
			ID nid = p->locate(i)->getID();
			const vector<Synapse *> &s_vec = p->locate(i)->getSynapses();
			for (int delay_t=0; delay_t < deltaDelay; delay_t++) {
				ret->pConnection->pDelayStart[delay_t + deltaDelay*nid] = synapseIdx;

				for (auto iter = s_vec.begin(); iter != s_vec.end(); iter++) {
					if ((*iter)->getDelaySteps(info.dt) == delay_t + minDelaySteps) {
						int sid = (*iter)->getID();
						assert(synapseIdx < totalSynapseNum);
						assert(synapseIdx == sid);
						synapseIdx++;
					}
				}

				ret->pConnection->pDelayNum[delay_t + deltaDelay*nid] = synapseIdx - ret->pConnection->pDelayStart[delay_t + deltaDelay*nid];
			}
		}
	}

	return ret;
}
