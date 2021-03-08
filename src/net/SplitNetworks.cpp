
#include "Network.h"

#define SYN_BASE 0
#define NEU_BASE 1
#define ROUND_ROBIN 2
#define BALANCED 3

#define SPLIT ROUND_ROBIN 

void Network::splitNetwork()
{
#if SPLIT==NEU_BASE
	printf("===========NEU_BASE==========\n");
	int node_idx = 0;
	size_t neuron_count = 0;
	size_t neuron_per_node = _neuron_num/_node_num;
	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_iter->first;
		for (size_t i=0; i<t_iter->second.size(); i++) {
			ID id(t, 0, i);
			_nid2node[id] = node_idx;
			neuron_count += 1;
			for (auto iter = n2s_conn_rev[id].begin(); iter != n2s_conn_rev[id].end(); iter++) {
				for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
					_sid2node[*siter] = node_idx;
				}
			}
			if (node_count>= (node_idx+1) * neuron_per_node && node_idx < _node_num - 1) {
				node_idx++;	
			}
		}
	}
#elif SPLIT==ROUND_ROBIN
	printf("===========ROUND_ROBIN==========\n");
	size_t neuron_count = 0;
	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_iter->first;
		for (size_t i=0; i<t_iter->second.size(); i++) {
			ID id(t, 0, i);
			int node_idx = neuron_count % _node_num;
			_nid2node[id] = node_idx;
			neuron_count += 1;
			for (auto iter = n2s_conn_rev[id].begin(); iter != n2s_conn_rev[id].end(); iter++) {
				for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
					_sid2node[*siter] = node_idx;
				}
			}
		}
	}
#elif SPLIT==BALANCED
	printf("===========BALANCED==========\n");
#else
	printf("===========SYN_BASE==========\n");
	int node_idx = 0;
	size_t synapse_count = 0;
	size_t synapse_per_node = _synapse_num/_node_num;
	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_ter->first;
		for (size_t i=0; i<t_iter->second.size(); i++) {
			ID id(t, 0, i);
			_nid2node[id] = node_idx;
			for (auto iter = n2s_conn_rev[id].begin(); iter != n2s_conn_rev[id].end(); iter++) {
				for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
					_sid2node[*siter] = node_idx;
				}
				synapse_count += iter->second.size();
			}
			if (synapse_count >= (node_idx+1) * synapse_per_node && node_idx < _node_num - 1) {
				node_idx++;	
			}
		}
	}
#endif

	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_iter->first;
		for (size_t i=0; i<t_iter->second.size(); i++) {
			bool cross_node = false;
			ID id(t, 0, i);
			unsigned int n_node = _nid2node[id];
			for (auto iter = n2s_conn[id].begin(); iter != n2s_conn[id].end(); iter++) {
				for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
					unsigned int s_node = _sid2node[*siter];
					if (n_node != s_node) {
						cross_node = true;
						_crossnodeNeuronsRecv[s_node].insert(id);
					}
				}
			}
			if (cross_node) {
				_crossnodeNeuronsSend[n_node].insert(id);
			}
		}
	}

	return;
}

