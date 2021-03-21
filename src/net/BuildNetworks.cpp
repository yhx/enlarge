
#include "../base/TypeFunc.h"
#include "../utils/proc_info.h"
#include "../utils/utils.h"
#include "../../include/Synapses.h"
#include "Network.h"
#include "Network.h"

void Network::update_status_splited()
{
	_neuron_nums.clear();
	_synapse_nums.clear();
	_crossnodeNeuronsRecv.clear();
	_crossnodeNeuronsSend.clear();
	_crossnodeNeuronsSend.resize(_node_num);
	_crossnodeNeuronsRecv.resize(_node_num);
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

int Network::arrangeNeuron(DistriNetwork *net, CrossTypeInfo_t &type_offset, CrossTypeInfo_t &neuron_offset, CrossTypeInfo_t &neuron_count)
{
	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
			Type t = t_iter->first;
			for (size_t i=0; i<t_iter->second->size(); i++) {
				ID id(t, 0, i);
				unsigned int n_node = _nid2node[id];
				_neurons[t]->packup(net[n_node]._network->ppNeurons[type_offset[n_node][t]], neuron_count[n_node][t], i);
				_id2node_idx[id] = neuron_offset[n_node][t] + neuron_count[n_node][t];
				neuron_count[n_node][t]++;
			}
	}

	return 0;
}

int Network::arrangeLocal(DistriNetwork *net, CrossTypeInfo_t &type_offset, CrossTypeInfo_t &neuron_offset, CrossTypeInfo_t & synapse_offset, CrossTypeInfo_t &neuron_count, CrossTypeInfo_t &synapse_count, CrossTypeInfo_t &n2s_count, map<unsigned int, size_t> &n_num)
{

	for (unsigned int d=_min_delay; d<_max_delay+1; d++) {
		for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
			Type t = t_iter->first;
			for (size_t i=0; i<t_iter->second->size(); i++) {
				ID id(t, 0, i);
				unsigned int n_node = _nid2node[id];
				size_t n_offset = (n_num[n_node]+_crossnodeNeuronsRecv[n_node].size())*(d-_min_delay)+_id2node_idx[id];
				for (auto siter = n2s_conn[id][d].begin(); siter != n2s_conn[id][d].end(); siter++) {
					unsigned int s_node = _sid2node[*siter];
					if (s_node == n_node) {
						Type s_t = siter->type();
						unsigned int s_idx = type_offset[s_node][s_t];
						_id2node_idx[*siter] = synapse_count[s_node][s_t];
						_synapses[s_t]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][s_t], siter->id());
						Connection * c = net[s_node]._network->ppConnections[s_idx];
						c->pDelayStart[n_offset] = n2s_count[s_node][s_t];
						c->pSidMap[synapse_count[s_node][s_t]] = synapse_count[s_node][s_t];
						synapse_count[s_node][s_t]++;
					}
				}
				for (auto s_t = _synapse_nums[n_node].begin(); s_t != _synapse_nums[n_node].end(); s_t++) {
					unsigned int s_idx = type_offset[n_node][s_t->first];
					Connection * c = net[n_node]._network->ppConnections[s_idx];
					c->pDelayNum[n_offset] = synapse_count[n_node][s_t->first] - n2s_count[n_node][s_t->first];
					c->pDelayStart[n_offset+1] = c->pDelayStart[n_offset] + c->pDelayNum[n_offset];
					n2s_count[n_node][s_t->first] = synapse_count[n_node][s_t->first];
				}

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

int Network::arrangeCross(DistriNetwork *net, CrossTypeInfo_t & type_offset, CrossTypeInfo_t &synapse_count, CrossTypeInfo_t &n2s_count, map<unsigned int, size_t> &n_num)
{
	map<unsigned int, size_t> cross_idx; // idx for cross-node neuron
	map<unsigned int, size_t> node_n_offset; // neuron offset for each node
	vector<bool> cross_nodes(_node_num, false);

	for (auto d=_min_delay; d<_max_delay+1; d++) {
		for (int n_node = 0; n_node < _node_num; n_node++) {
			CrossNodeMap *map = net[n_node]._crossnodeMap;
			for (auto iter = _crossnodeNeuronsSend[n_node].begin(); iter != _crossnodeNeuronsSend[n_node].end(); iter++) {
				assert(n_node == _nid2node[*iter]);
				if (d == _min_delay) {
					assert(cross_idx[n_node] < _crossnodeNeuronsSend[n_node].size());
					map->_idx2index[_id2node_idx[*iter]] = cross_idx[n_node];
					cross_idx[n_node]++;
				}

				fill(cross_nodes.begin(), cross_nodes.end(), false);
				for (auto siter = n2s_conn[*iter][d].begin(); siter != n2s_conn[*iter][d].end(); siter++) {
					int s_node = _sid2node[*siter];
					if (s_node != n_node) {
						cross_nodes[s_node] = true;
						assert(map->_idx2index[_id2node_idx[*iter]] >= 0);
					    auto index_t =  map->_idx2index[_id2node_idx[*iter]] * _node_num + s_node;
						assert(map->_crossnodeIndex2idx != NULL);
						auto idx_t = map->_crossnodeIndex2idx[index_t];
						if ( idx_t < 0) {
							idx_t = n_num[s_node] + node_n_offset[s_node];
							map->_crossnodeIndex2idx[index_t] = idx_t;
							node_n_offset[s_node]++;
						} 
						assert(idx_t >= 0);

						size_t n_offset = (n_num[s_node]+_crossnodeNeuronsRecv[s_node].size())*(d-_min_delay)+idx_t;
						Type s_t = siter->type();
						size_t s_idx = type_offset[s_node][s_t];
						_synapses[s_t]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][s_t], siter->id());
						Connection * c = net[s_node]._network->ppConnections[s_idx];
						c->pDelayStart[n_offset] = n2s_count[s_node][s_t];
						c->pSidMap[synapse_count[s_node][s_t]] = synapse_count[s_node][s_t];
						synapse_count[s_node][s_t]++;
					}
				}
				for (int node_t = 0; node_t < _node_num; node_t++) {
					if (node_t == n_node) {
						continue;
					}
					if (cross_nodes[node_t]) {
						for (auto s_i = _synapse_nums[node_t].begin(); s_i != _synapse_nums[node_t].end(); s_i++) {
							Type s_t = s_i->first;
							auto index_t =  map->_idx2index[_id2node_idx[*iter]] * _node_num + node_t;
							auto idx_t = map->_crossnodeIndex2idx[index_t];
							size_t n_offset = (n_num[node_t]+_crossnodeNeuronsRecv[node_t].size())*(d-_min_delay)+idx_t;
							Connection *c= net[node_t]._network->ppConnections[type_offset[node_t][s_t]];
							c->pDelayNum[n_offset] = synapse_count[node_t][s_t] - n2s_count[node_t][s_t];
							c->pDelayStart[n_offset+1] = c->pDelayStart[n_offset] + c->pDelayNum[n_offset];
							n2s_count[node_t][s_t] = synapse_count[node_t][s_t];
						}
					}
				}
			}
		}
	}

	for (int  node_t = 0; node_t < _node_num; node_t++) {
		assert(cross_idx[node_t] == _crossnodeNeuronsSend[node_t].size());
		assert(node_n_offset[node_t] == _crossnodeNeuronsRecv[node_t].size());
		for (auto i = _synapse_nums[node_t].begin(); i != _synapse_nums[node_t].end(); i++) {
			assert(synapse_count[node_t][i->first] == i->second);
		}
	}
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

	CrossTypeInfo_t type_offset;
	CrossTypeInfo_t neuron_offset;
	CrossTypeInfo_t synapse_offset;
	CrossTypeInfo_t neuron_count;
	CrossTypeInfo_t synapse_count;
	CrossTypeInfo_t n2s_count;

	map<unsigned int, size_t> s_num;
	map<unsigned int, size_t> n_num;


	for (int i=0; i<_node_num; i++) {
		net[i]._network = allocGNetwork(_neuron_nums[i].size(), _synapse_nums[i].size());
		size_t offset = 0;
		size_t e_offset = 0;
		for (auto n=_neuron_nums[i].begin(); n!=_neuron_nums[i].end(); n++) {
			net[i]._network->pNTypes[offset] = n->first;
			net[i]._network->pNeuronNums[offset] = e_offset;
			net[i]._network->ppNeurons[offset] = allocType[n->first](n->second);
			assert(net[i]._network->ppNeurons[offset] != NULL);
			neuron_count[i][n->first] = 0;
			type_offset[i][n->first] = offset;
			neuron_offset[i][n->first] = e_offset;
			offset++;
			e_offset += n->second;
		}
		net[i]._network->pNeuronNums[offset] = e_offset;
		n_num[i] = e_offset;

		offset = 0;
		e_offset = 0;
		for (auto s=_synapse_nums[i].begin(); s!=_synapse_nums[i].end(); s++) {
			net[i]._network->pSTypes[offset] = s->first;
			net[i]._network->pSynapseNums[offset] = e_offset;
			net[i]._network->ppSynapses[offset] = allocType[s->first](s->second);
			assert(net[i]._network->ppSynapses[offset] != NULL);
			net[i]._network->ppConnections[offset] = allocConnection(n_num[i] +  _crossnodeNeuronsRecv[i].size(), s->second, _max_delay, _min_delay);
			synapse_count[i][s->first] = 0;
			type_offset[i][s->first] = offset;
			synapse_offset[i][s->first] = e_offset;
			offset++;
			e_offset += s->second;
		}
		net[i]._network->pSynapseNums[offset] = e_offset;
		s_num[i] = e_offset;

		net[i]._crossnodeMap = allocCNM(n_num[i] + _crossnodeNeuronsRecv[i].size(), _crossnodeNeuronsSend[i].size(), _node_num);
	}

	printf("===================Arrange Local Data==========================\n");
	arrangeLocal(net, type_offset, neuron_offset, synapse_offset, neuron_count, synapse_count, n2s_count, n_num);

	printf("=================Arrange CrossNode Data========================\n");
	arrangeCross(net, type_offset, synapse_count, n2s_count, n_num);

	print_mem("Finish build");

	return net;
}
