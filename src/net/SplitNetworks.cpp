
#include "Network.h"

void Network::splitNetwork(SplitType split)
{
	switch (split) {
		case NeuronBalance:
			{
				printf("===========NEU_BASE==========\n");
				int node_idx = 0;
				size_t neuron_count = 0;
				size_t neuron_per_node = _neuron_num/_node_num;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);
						_nid2node[id] = node_idx;
						neuron_count += 1;
						for (auto iter = n2s_conn_rev[id].begin(); iter != n2s_conn_rev[id].end(); iter++) {
							for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
								_sid2node[*siter] = node_idx;
							}
						}
						if (neuron_count>= (node_idx+1) * neuron_per_node && node_idx < _node_num - 1) {
							node_idx++;	
						}
					}
				}
			}
			break;
		case RoundRobin:
			{
				printf("===========ROUND_ROBIN==========\n");
				size_t neuron_count = 0;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
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
			}
			break;
		case Balanced:
			{
				printf("===========BALANCED==========\n");
			}
			break;
		default:
			{
				printf("===========SYN_BASE==========\n");
				int node_idx = 0;
				size_t synapse_count = 0;
				size_t synapse_per_node = _synapse_num/_node_num;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
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
			}
	}

	return;
}

