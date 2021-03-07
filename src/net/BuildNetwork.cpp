
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>

#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "Network.h"

void Network::updateStatus()
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
	}
};

GNetwork* Network::buildNetwork(const SimInfo &info)
{
	struct sysinfo sinfo;
	sysinfo(&sinfo);
	printf("Before build, MEM used: %lfGB\n", static_cast<double>((sinfo.totalram - sinfo.freeram)/1024.0/1024.0/1024.0));

	updateStatus();

	size_t n_type_num = _neurons.size();
	size_t s_type_num = _synapses.size();
	// int delta_delay = _max_delay - _min_delay + 1;

	GNetwork * ret = allocGNetwork(n_type_num, s_type_num);

	int i=0;
	for (auto iter=_neurons.begin(); iter!=_neurons.end(); iter++) {
		ret->pNTypes[i] = iter->first;
		ret->ppNeurons[i] = iter->second->packup();
		assert(ret->ppNeurons[i] != NULL);
		ret->pNeuronNums[i+1] = iter->second->size() + ret->pNeuronNums[i];
	}
	assert(ret->pNeuronNums[n_type_num] == _neuron_num);

	map<Type, size_t> tp2idx;
	int idx = 0;
	for (auto iter=_synapses.begin(); iter!=_synapses.end(); iter++) {
		ret->pSTypes[i] = iter->first;
		ret->ppSynapses[i] = iter->second->packup();
		assert(ret->ppSynapses[i] != NULL);
		ret->pSynapseNums[i+1] = iter->second->size() + ret->pSynapseNums[i];
		tp2idx[iter->first] = idx;
		idx++;
	}
	assert(ret->pSynapseNums[n_type_num] == _synapse_num);

	ret->ppConnections = (Connection **)malloc(sizeof(Connection*)*s_type_num); 
	for (size_t i=0; i<s_type_num; i++) {
		ret->ppConnections[i] = allocConnection(ret->pNeuronNums[n_type_num], ret->pSynapseNums[i+1] - ret->pSynapseNums[i], _max_delay, _min_delay);
	}

	size_t *syn_idx = (size_t *)malloc(sizeof(size_t) * s_type_num); 
	size_t *count = (size_t *)malloc(sizeof(size_t) * s_type_num); 
	memset(syn_idx, 0, sizeof(size_t)*s_type_num);
	for (size_t i=0; i<n_type_num; i++) {
		Type t = ret->pNTypes[i];
		for (size_t n=0; n<ret->pNeuronNums[i+1]-ret->pNeuronNums[i]; n++) {
			ID nid(t, 0, n);
			for (auto d_iter = n2s_conn[nid].begin(); d_iter != n2s_conn[nid].end(); d_iter++) {
				unsigned int d = d_iter->first;
				size_t n_offset = _neuron_num*d+_neurons_offset[t]+n;
				memset(count, 0, sizeof(size_t)*s_type_num);
				for (auto s_iter = d_iter->second.begin(); s_iter != d_iter->second.end(); s_iter++) {
					int idx = tp2idx[s_iter->type()];
					Connection * c = ret->ppConnections[idx];
					c->pDelayStart[n_offset] = syn_idx[idx];
					c->pSidMap[syn_idx[idx]] = s_iter->id();
					syn_idx[idx]++;
					count[idx]++;
				}
				assert(syn_idx[idx] - ret->ppConnections[idx]->pDelayStart[n_offset] == count[idx]);
				for (size_t s=0; s<s_type_num; s++) {
					ret->ppConnections[s]->pDelayNum[n_offset] = count[idx];
				}

			}
		}
	}

	sysinfo(&sinfo);
	printf("Finish build, MEM used: %lfGB\n", static_cast<double>((sinfo.totalram - sinfo.freeram)/1024.0/1024.0/1024.0));

	return ret;
}
