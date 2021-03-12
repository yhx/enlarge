
#include "../utils/TypeFunc.h"
#include "../utils/proc_info.h"
#include "../utils/utils.h"
#include "../../include/Synapses.h"
#include "Network.h"
#include "Network.h"

void Network::update_status_splited()
{
	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_iter->first;
		for (size_t i=0; i<t_iter->second->size(); i++) {
			bool cross_node = false;
			ID id(t, 0, i);
			unsigned n_node = _nid2node[id];
			_neuron_nums[n_node][t] += 1;
			for (auto iter = n2s_conn[id].begin(); iter != n2s_conn[id].end(); iter++) {
				for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
					unsigned int s_node = _sid2node[*siter];
					_synapse_nums[s_node][siter->type()] += 1;
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
}


int Network::arrangeLocal(DistriNetwork *net, CrossTypeInfo_t & type_offset, CrossTypeInfo_t &neuron_offset, CrossTypeInfo_t & synapse_offset, CrossTypeInfo_t &neuron_count, CrossTypeInfo_t &synapse_count, CrossTypeInfo_t &n2s_count, map<unsigned int, size_t> &n_num)
{
	printf("===================Arrange Local Data==========================\n");
	for (unsigned int d=0; d<_max_delay-_min_delay+1; d++) {
		for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
			Type t = t_iter->first;
			for (size_t i=0; i<t_iter->second->size(); i++) {
				ID id(t, 0, i);
				unsigned n_node = _nid2node[id];
				_neurons[t]->packup(net[n_node]._network->ppNeurons[type_offset[n_node][t]], neuron_count[n_node][t], i);
				_id2node_idx[id] = neuron_offset[n_node][t] + neuron_count[n_node][t];
				size_t n_offset = (n_num[n_node]+_crossnodeNeuronsRecv[n_node].size())*d+neuron_offset[n_node][t]+neuron_count[n_node][t];
				for (auto siter = n2s_conn[id][d].begin(); siter != n2s_conn[id][d].end(); siter++) {
					unsigned int s_node = _sid2node[*siter];
					if (s_node == n_node) {
						Type s_t = siter->type();
						unsigned int s_idx = type_offset[s_node][s_t];
						_id2node_idx[*siter] = synapse_count[s_node][s_t];
						_synapses[t]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][s_t], siter->id());
						Connection * c = net[s_node]._network->ppConnections[s_idx];
						c->pDelayStart[n_offset] = n2s_count[s_node][t];
						c->pSidMap[synapse_count[s_node][s_t]] = synapse_count[s_node][s_t];
						synapse_count[s_node][s_t]++;
					}
				}
				for (auto s_t = _synapse_nums[n_node].begin(); s_t != _synapse_nums[n_node].end(); s_t++) {
					unsigned int s_idx = type_offset[n_node][s_t->first];
					net[n_node]._network->ppConnections[s_idx]->pDelayNum[n_offset] = synapse_count[n_node][s_t->first] - n2s_count[n_node][t];
					n2s_count[n_node][s_t->first] = synapse_count[n_node][s_t->first];
				}

				neuron_count[n_node][t]++;
			}
		}
	}

	for (auto node = _neuron_nums.begin(); node != _neuron_nums.end(); node++) {
		for (auto t = node->second.begin(); t != node->second.end(); t++) {
			assert(neuron_count[node->first][t->first] == t->second);
		}
	}

	return 0;
}

int Network::arrangeCross(DistriNetwork *net, CrossTypeInfo_t & type_offset, CrossTypeInfo_t & neuron_offset, CrossTypeInfo_t &synapse_count, CrossTypeInfo_t &n2s_count, map<unsigned int, size_t> &n_num)
{
	printf("=================Arrange CrossNode Data========================\n");
	map<unsigned int, size_t> node_offset;
	map<unsigned int, size_t> cross_offset;
	vector<bool> cross_nodes(_node_num, false);
	for (unsigned int d=0; d<_max_delay-_min_delay+1; d++) {
		for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
			Type t = t_iter->first;
			for (size_t i=0; i<t_iter->second->size(); i++) {
				ID id(t, 0, i);
				unsigned n_node = _nid2node[id];

				if (_crossnodeNeuronsSend[n_node].find(id) == _crossnodeNeuronsSend[n_node].end()) {
					continue;
				} else {
					net[n_node]._crossnodeMap->_idx2index[_id2node_idx[id]] = node_offset[n_node];
					fill(cross_nodes.begin(), cross_nodes.end(), false);
					for (auto siter = n2s_conn[id][d].begin(); siter != n2s_conn[id][d].end(); siter++) {
						unsigned int s_node = _sid2node[*siter];
						if (s_node != n_node) {
							cross_nodes[s_node] = true;
							size_t n_offset = (n_num[s_node]+_crossnodeNeuronsRecv[s_node].size())*d+n_num[s_node]+cross_offset[s_node];
							Type s_t = siter->type();
							unsigned int s_idx = type_offset[s_node][s_t];
							_synapses[t]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][s_t], siter->id());
							Connection * c = net[s_node]._network->ppConnections[s_idx];
							c->pDelayStart[n_offset] = n2s_count[s_node][t];
							c->pSidMap[synapse_count[s_node][s_t]] = synapse_count[s_node][s_t];
							synapse_count[s_node][s_t]++;
						}
					}
					for (unsigned int  n_t = 0; n_t < _node_num; n_t++) {
						if (cross_nodes[n_t]) {
							for (auto s_t = _synapse_nums[n_t].begin(); s_t != _synapse_nums[n_t].end(); s_t++) {
								size_t n_offset = (n_num[n_t]+_crossnodeNeuronsRecv[n_t].size())*d+neuron_offset[n_t][t]+neuron_count[n_t][t];
								unsigned int s_idx = type_offset[n_node][s_t];
								net[n_node]._network->ppConnections[s_idx]->pDelayNum[n_offset] = synapse_count[n_node][s_t] - n2s_count[n_node][t];
								n2s_count[n_node][s_t] = synapse_count[n_node][s_t];
							}
							net[n_t]._crossnodeMap->_crossnodeIndex2idx[node_offset[n_t]*_node_num+n_t] = neuron_offset[n_t][t] + neuron_count[n_t][t];
							cross_offset[n_t]++;
						}
					}

					node_offset[n_node]++;
				}
			}
		}
	}

	for (unsigned int  n_t = 0; n_t < _node_num; n_t++) {
		assert(cross_offset[n_t] == _crossnodeNeuronsRecv[n_t]);
		for (auto t = _synapse_nums[n_t].begin(); t != _synapse_nums[n_t].end(); t++) {
			assert(synapse_count[n_t][t] == t->second);
		}
	}
	return 0;
}

int Network::arrangeNet(DistriNetwork *net) 
{
	CrossTypeInfo_t type_offset;
	CrossTypeInfo_t neuron_offset;
	CrossTypeInfo_t synapse_offset;
	CrossTypeInfo_t neuron_count;
	CrossTypeInfo_t synapse_count;
	CrossTypeInfo_t n2s_count;

	map<unsigned int, size_t> s_num;
	map<unsigned int, size_t> n_num;


	for (unsigned int i=0; i<_node_num; i++) {
		net[i]._network = allocGNetwork(_neuron_nums[i].size(), _synapse_nums[i].size());
		size_t offset = 0;
		size_t e_offset = 0;
		for (auto n=_neuron_nums[i].begin(); n!=_neuron_nums[i].end(); n++) {
			net[i]._network->pNTypes[offset] = n->first;
			net[i]._network->ppNeurons[offset] = allocType[n->first](n->second);
			neuron_count[i][n->first] = 0;
			type_offset[i][n->first] = offset;
			neuron_offset[i][n->first] = e_offset;
			offset++;
			e_offset += n->second;
		}
		n_num[i] = e_offset;

		offset = 0;
		e_offset = 0;
		for (auto s=_synapse_nums[i].begin(); s!=_synapse_nums[i].end(); s++) {
			net[i]._network->pSTypes[offset] = s->first;
			net[i]._network->ppSynapses[offset] = allocType[s->first](s->second);
			synapse_count[i][s->first] = 0;
			type_offset[i][s->first] = offset;
			synapse_offset[i][s->first] = e_offset;
			offset++;
			e_offset += s->second;
		}
		s_num[i] = e_offset;

		net[i]._crossnodeMap = allocCNM(n_num[i] + _crossnodeNeuronsRecv[i].size(), _crossnodeNeuronsSend[i].size(), _node_num);
	}

	arrangeLocal(net, type_offset, neuron_offset, synapse_offset, neuron_count, synapse_count, n2s_count, n_num);

	arrangeCross(net, synapse_count, n2s_count, n_num);



	return 0;
}

DistriNetwork* Network::buildNetworks(const SimInfo &info, bool auto_splited)
{
	print_mem("Before build");
	printf("===================Update Status================================\n");
	update_status();

	assert(_node_num >= 1);
	DistriNetwork * net = initDistriNet(_node_num, _dt);

	printf("=====================Split Network=============================\n");
	if (auto_splited && _node_num > 1) {
		splitNetwork();
	}

	update_status_splited();

	for (int nodeIdx =0; nodeIdx <_node_num; nodeIdx++) {
		net[nodeIdx]._network = arrangeData(nodeIdx, info);

		int nNum = net[nodeIdx]._network->pNeuronNums[net[nodeIdx]._network->nTypeNum] + _crossnodeNeuron2idx[nodeIdx].size();
		int sNum = net[nodeIdx]._network->pSynapseNums[net[nodeIdx]._network->sTypeNum];
		net[nodeIdx]._network->pConnection = arrangeConnect(nNum, sNum, nodeIdx, info);

	}

	print_mem("After build");

	printf("=====================Arrange Map===============================\n");
	for (int nodeIdx =0; nodeIdx <_node_num; nodeIdx++) {
		int nNum = net[nodeIdx]._network->pNeuronNums[net[nodeIdx]._network->nTypeNum] + _crossnodeNeuron2idx[nodeIdx].size();
		net[nodeIdx]._crossnodeMap = arrangeCrossNodeMap(nNum, nodeIdx, _node_num);
	}

	print_mem("Finish build");

	if (info.save_mem) {
		for (auto piter = _pPopulations.begin(); piter != _pPopulations.end(); piter++) {
			Population * p = *piter;
			for (auto niter = p->_items.begin(); niter != p->_items.end(); niter++) {
				Neuron *n = *niter;
				vector<Synapse *> &s_vec = n->getSynapses();
				for (auto siter = s_vec.begin(); siter != s_vec.end(); siter++) {
					Synapse *s = *siter;
					delete s;
					*siter = NULL;
				}
				delete n;
				*niter = NULL;
			}
			delete p;
			*piter = NULL;
		}

		print_mem("Save MEM");
	}

	return net;
}
