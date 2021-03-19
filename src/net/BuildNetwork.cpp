
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <algorithm>

#include "../utils/helper_c.h"
#include "../utils/utils.h"
#include "../base/TypeFunc.h"
#include "Network.h"

void Network::update_status()
{
	_neuron_num = 0;
	for (auto iter=_neurons.begin(); iter!=_neurons.end(); iter++) {
		_neurons_offset[iter->first] = _neuron_num;
		_neuron_num += iter->second->size();
	}

	_synapse_num = 0;
	for (auto iter=_synapses.begin(); iter!=_synapses.end(); iter++) {
		_synapses_offset[iter->first] = _synapse_num;
		_synapse_num += iter->second->size();

		auto t = minmax_element(iter->second->delay().begin(), iter->second->delay().end());
		unsigned int max = *(t.second);
		unsigned int min = *(t.first);
		if (_max_delay < max) {
			_max_delay = max;
		}
		if (_min_delay > min) {
			_min_delay = min;
		}
	}
	assert(_min_delay <= _max_delay);
}

GNetwork* Network::buildNetwork(const SimInfo &info)
{
	struct sysinfo sinfo;
	sysinfo(&sinfo);
	printf("Before build, MEM used: %lfGB\n", static_cast<double>((sinfo.totalram - sinfo.freeram)/1024.0/1024.0/1024.0));

	update_status();

	size_t n_type_num = _neurons.size();
	size_t s_type_num = _synapses.size();
	// int delta_delay = _max_delay - _min_delay + 1;

	GNetwork * ret = allocGNetwork(n_type_num, s_type_num);

	int n_t=0;
	for (auto iter=_neurons.begin(); iter!=_neurons.end(); iter++) {
		ret->pNTypes[n_t] = iter->first;
		ret->ppNeurons[n_t] = iter->second->packup();
		assert(ret->ppNeurons[n_t] != NULL);
		ret->pNeuronNums[n_t+1] = iter->second->size() + ret->pNeuronNums[n_t];
		n_t++;
	}
	assert(ret->pNeuronNums[n_type_num] == _neuron_num);

	map<Type, size_t> tp2idx;
	int s_t = 0;
	for (auto iter=_synapses.begin(); iter!=_synapses.end(); iter++) {
		ret->pSTypes[s_t] = iter->first;
		ret->ppSynapses[s_t] = iter->second->packup();
		assert(ret->ppSynapses[s_t] != NULL);
		ret->pSynapseNums[s_t+1] = iter->second->size() + ret->pSynapseNums[s_t];
		tp2idx[iter->first] = s_t;
		s_t++;
	}
	assert(ret->pSynapseNums[n_type_num] == _synapse_num);

	ret->ppConnections = malloc_c<Connection*>(s_type_num); 
	for (size_t i=0; i<s_type_num; i++) {
		ret->ppConnections[i] = allocConnection(ret->pNeuronNums[n_type_num], ret->pSynapseNums[i+1] - ret->pSynapseNums[i], _max_delay, _min_delay);
	}

	size_t *syn_idx = malloc_c<size_t>(s_type_num); 
	size_t *start = malloc_c<size_t>(s_type_num); 
	for (unsigned int d=_min_delay; d<_max_delay+1; d++) {
		for (size_t i=0; i<n_type_num; i++) {
			Type t = ret->pNTypes[i];
			for (size_t n=0; n<ret->pNeuronNums[i+1]-ret->pNeuronNums[i]; n++) {
				ID nid(t, 0, n);
				size_t n_offset = _neuron_num*(d-_min_delay) + _neurons_offset[t]+n;
				for (auto s_iter = n2s_conn[nid][d].begin(); s_iter != n2s_conn[nid][d].end(); s_iter++) {
					int s_idx = tp2idx[s_iter->type()];
					Connection * c = ret->ppConnections[s_idx];
					c->pDelayStart[n_offset] = start[s_idx];
					c->pSidMap[syn_idx[s_idx]] = s_iter->id();
					syn_idx[s_idx]++;
				}
				for (size_t s=0; s<s_type_num; s++) {
					ret->ppConnections[s]->pDelayNum[n_offset] = syn_idx[s] - start[s];
					ret->ppConnections[s]->pDelayStart[n_offset+1] = ret->ppConnections[s]->pDelayStart[n_offset] +  ret->ppConnections[s]->pDelayNum[n_offset];
					start[s] = syn_idx[s];
				}
			}
		}
	}

	for (size_t st=0; st<s_type_num; st++) {
		for (unsigned int i=0; i<ret->pNeuronNums[n_type_num] * (_max_delay - _min_delay+1); i++) {
			assert(ret->ppConnections[st]->pDelayStart[i+1] == ret->ppConnections[st]->pDelayStart[i] + ret->ppConnections[st]->pDelayNum[i]);
		}
	}

	sysinfo(&sinfo);
	printf("Finish build, MEM used: %lfGB\n", static_cast<double>((sinfo.totalram - sinfo.freeram)/1024.0/1024.0/1024.0));

	return ret;
}
