
#include <assert.h>
#include <sys/sysinfo.h>

#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "Network.h"


GNetwork* Network::buildNetwork(const SimInfo &info)
{
	struct sysinfo sinfo;
	sysinfo(&sinfo);
	printf("Before build, MEM used: %lfGB\n", static_cast<double>((sinfo.totalram - sinfo.freeram)/1024.0/1024.0/1024.0));

	size_t n_type_num = _neurons.size();
	size_t s_type_num = _synapses.size();
	int delta_elay = _max_delay - _min_delay + 1;

	GNetwork * ret = allocGNetwork(n_type_num, s_type_num);

	int i=0;
	for (auto iter=_neurons.begin(); iter!=_neurons.end(); iter++) {
		ret->pNTypes[i] = iter->first;

		ret->ppNeurons[i] = iter->second->packup();
		assert(ret->ppNeurons[i] != NULL);
		ret->pNeuronNums[i+1] = iter->second->size() + ret->pNeuronNums[i];
	}

	for (auto iter=_synapses.begin(); iter!=_synapses.end(); iter++) {
		ret->pSTypes[i] = iter->first;

		ret->ppSynapses[i] = iter->second->packup();
		assert(ret->ppSynapses[i] != NULL);
		ret->pSynapseNums[i+1] = iter->second->size() + ret->pSynapseNums[i];
	}

	ret->pConnection = allocConnection(ret->pNeuronNums[n_type_num], ret->pSynapseNums[s_type_num], _max_delay, _min_delay);

	size_t synapseIdx = 0;
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
		Population * p = *pIter;
		for (size_t i=0; i<p->getNum(); i++) {
			ID nid = p->locate(i)->getID();
			const vector<Synapse *> &s_vec = p->locate(i)->getSynapses();
			for (int delay_t=0; delay_t < deltaDelay; delay_t++) {
				ret->pConnection->pDelayStart[delay_t + deltaDelay*nid] = synapseIdx;

				for (auto iter = s_vec.begin(); iter != s_vec.end(); iter++) {
					if ((*iter)->getDelaySteps(info.dt) == delay_t + _min_delay) {
						ID sid = (*iter)->getID();
						assert(synapseIdx < _totalSynapseNum);
						assert(synapseIdx == sid);
						synapseIdx++;
					}
				}

				ret->pConnection->pDelayNum[delay_t + deltaDelay*nid] = synapseIdx - ret->pConnection->pDelayStart[delay_t + deltaDelay*nid];
			}
		}
	}

	sysinfo(&sinfo);
	printf("Finish build, MEM used: %lfGB\n", static_cast<double>((sinfo.totalram - sinfo.freeram)/1024.0/1024.0/1024.0));

	return ret;
}
