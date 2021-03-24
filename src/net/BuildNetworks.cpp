
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

	_buffer_offsets.clear();
	for (int node=0; node<_node_num; node++) {
		size_t count = 0;
		for (auto iter=_neuron_nums[node].begin(); iter!=_neuron_nums[node].end(); iter++) {
			_buffer_offsets[node][iter->first] = count;
			count += iter->second * _neurons[iter->first]->buffer_size();
		}
	}
}

int Network::arrangeNet(DistriNetwork *net, CrossTypeInfo_t &type_offset, CrossTypeInfo_t &neuron_offset, CrossTypeInfo_t &neuron_count, CrossTypeInfo_t &synapse_offset, CrossTypeInfo_t &synapse_count)
{
	for (int i=0; i<_node_num; i++) {
		net[i]._network = allocGNetwork(_neuron_nums[i].size(), _synapse_nums[i].size());
		size_t t_offset = 0;
		size_t n_offset = 0;
		for (auto n=_neuron_nums[i].begin(); n!=_neuron_nums[i].end(); n++) {
			net[i]._network->pNTypes[t_offset] = n->first;
			net[i]._network->pNeuronNums[t_offset] = n_offset;
			net[i]._network->ppNeurons[t_offset] = allocType[n->first](n->second);
			assert(net[i]._network->ppNeurons[t_offset] != NULL);
			neuron_count[i][n->first] = 0;
			type_offset[i][n->first] = t_offset;
			neuron_offset[i][n->first] = n_offset;
			t_offset++;
			n_offset += n->second;
		}
		net[i]._network->pNeuronNums[t_offset] = n_offset;
		// n_num[i] = n_offset;

		t_offset = 0;
		size_t s_offset = 0;
		for (auto s=_synapse_nums[i].begin(); s!=_synapse_nums[i].end(); s++) {
			net[i]._network->pSTypes[t_offset] = s->first;
			net[i]._network->pSynapseNums[t_offset] = s_offset;
			net[i]._network->ppSynapses[t_offset] = allocType[s->first](s->second);
			assert(net[i]._network->ppSynapses[t_offset] != NULL);
			net[i]._network->ppConnections[t_offset] = allocConnection(n_offset +  _crossnodeNeuronsRecv[i].size(), s->second, _max_delay, _min_delay);
			synapse_count[i][s->first] = 0;
			type_offset[i][s->first] = t_offset;
			synapse_offset[i][s->first] = s_offset;
			t_offset++;
			s_offset += s->second;
		}
		net[i]._network->pSynapseNums[t_offset] = s_offset;
		// s_num[i] = s_offset;

		net[i]._crossnodeMap = allocCNM(n_offset + _crossnodeNeuronsRecv[i].size(), _crossnodeNeuronsSend[i].size(), _node_num);
	}

	return 0;
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

int Network::arrangeLocal(DistriNetwork *net, CrossTypeInfo_t &type_offset, CrossTypeInfo_t &neuron_offset, CrossTypeInfo_t &synapse_offset, CrossTypeInfo_t &neuron_count, CrossTypeInfo_t &synapse_count, CrossTypeInfo_t &n2s_count, int delay)
{

	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_iter->first;
		for (size_t i=0; i<t_iter->second->size(); i++) {
			ID id(t, 0, i);
			unsigned int n_node = _nid2node[id];
		    size_t n_num = net[n_node]._network->pNeuronNums[_neuron_nums[n_node].size()];
			size_t n_offset = (n_num+_crossnodeNeuronsRecv[n_node].size())*(delay-_min_delay)+_id2node_idx[id];
			for (auto siter = n2s_conn[id][delay].begin(); siter != n2s_conn[id][delay].end(); siter++) {
				unsigned int s_node = _sid2node[*siter];
				if (s_node == n_node) {
					Type s_t = siter->type();
					unsigned int s_idx = type_offset[s_node][s_t];
					_id2node_idx[*siter] = synapse_count[s_node][s_t];
					_synapses[s_t]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][s_t], siter->id());
					Connection * c = net[s_node]._network->ppConnections[s_idx];
					c->pDelayStart[n_offset] = n2s_count[s_node][s_t];
					c->pSidMap[synapse_count[s_node][s_t]] = synapse_count[s_node][s_t];
					ID target = s2n_conn[*siter];
					c->dst[synapse_count[s_node][s_t]] = _buffer_offsets[s_node][target.type()] + target.offset() * _neuron_nums[s_node][target.type()] + _id2node_idx[target.mask_offset()];
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

	return 0;
}

int Network::arrangeCross(DistriNetwork *net, CrossTypeInfo_t & type_offset, CrossTypeInfo_t &synapse_count, CrossTypeInfo_t &n2s_count, CrossInfo_t &cross_idx, CrossInfo_t &node_n_offset, int delay)
{
	if (delay == _min_delay) {
		for (int n_node = 0; n_node < _node_num; n_node++) {
			CrossNodeMap *map = net[n_node]._crossnodeMap;
			for (auto iter = _crossnodeNeuronsSend[n_node].begin(); iter != _crossnodeNeuronsSend[n_node].end(); iter++) {
				assert(n_node == _nid2node[*iter]);
				assert(cross_idx[n_node] < _crossnodeNeuronsSend[n_node].size());
				map->_idx2index[_id2node_idx[*iter]] = cross_idx[n_node];
				cross_idx[n_node]++;
			}
		}
	}

	for (int s_node = 0; s_node < _node_num; s_node++) {
		for (auto iter = _crossnodeNeuronsRecv[s_node].begin(); iter != _crossnodeNeuronsRecv[s_node].end(); iter++) {
			int n_node = _nid2node[*iter];
			assert(s_node != n_node);
			CrossNodeMap *n_map = net[n_node]._crossnodeMap;

			assert(n_map->_idx2index[_id2node_idx[*iter]] >= 0);
			auto index_t = n_map->_idx2index[_id2node_idx[*iter]] * _node_num + s_node;
			assert(n_map->_crossnodeIndex2idx != NULL);

			size_t n_num = net[s_node]._network->pNeuronNums[_neuron_nums[s_node].size()];

			if (delay == _min_delay) {
				assert(n_map->_crossnodeIndex2idx[index_t] == -1);
				n_map->_crossnodeIndex2idx[index_t] = n_num + node_n_offset[s_node];
				node_n_offset[s_node]++;
			}

			auto idx_t = n_map->_crossnodeIndex2idx[index_t];
			size_t n_offset = (n_num+_crossnodeNeuronsRecv[s_node].size())*(delay-_min_delay)+idx_t;

			for (auto siter = n2s_conn[*iter][delay].begin(); siter != n2s_conn[*iter][delay].end(); siter++) {
				if (s_node == _sid2node[*siter]) {
					Type s_t = siter->type();
					size_t s_idx = type_offset[s_node][s_t];
					_synapses[s_t]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][s_t], siter->id());
					Connection * c = net[s_node]._network->ppConnections[s_idx];
					c->pDelayStart[n_offset] = n2s_count[s_node][s_t];
					c->pSidMap[synapse_count[s_node][s_t]] = synapse_count[s_node][s_t];
					ID target = s2n_conn[*siter];
					c->dst[synapse_count[s_node][s_t]] = _buffer_offsets[s_node][target.type()] + target.offset() * _neuron_nums[s_node][target.type()] + _id2node_idx[target.mask_offset()];
					synapse_count[s_node][s_t]++;
				}
			}
			for (auto s_i = _synapse_nums[s_node].begin(); s_i != _synapse_nums[s_node].end(); s_i++) {
				Type s_t = s_i->first;
				Connection *c= net[s_node]._network->ppConnections[type_offset[s_node][s_t]];
				c->pDelayNum[n_offset] = synapse_count[s_node][s_t] - n2s_count[s_node][s_t];
				c->pDelayStart[n_offset+1] = c->pDelayStart[n_offset] + c->pDelayNum[n_offset];
				n2s_count[s_node][s_t] = synapse_count[s_node][s_t];
			}
		}
	}

	return 0;
}

DistriNetwork* Network::buildNetworks(const SimInfo &info, SplitType split, bool auto_splited)
{
	print_mem("Before build");
	printf("===================Update Status================================\n");
	update_status();

	assert(_node_num >= 1);
	DistriNetwork * net = initDistriNet(_node_num, _dt);

	printf("=====================Split Network=============================\n");
	if (auto_splited && _node_num > 1) {
		splitNetwork(split);
	}

	update_status_splited();

	CrossTypeInfo_t type_offset;
	CrossTypeInfo_t neuron_offset;
	CrossTypeInfo_t synapse_offset;
	CrossTypeInfo_t neuron_count;
	CrossTypeInfo_t synapse_count;
	CrossTypeInfo_t n2s_count;

	CrossInfo_t cross_idx; // idx for cross-node neuron
	CrossInfo_t node_n_offset; // neuron offset for each node

	// CrossInfo_t s_num;
	// CrossInfo_t n_num;


	printf("=====================Arrange Networ============================\n");
	arrangeNet(net, type_offset, neuron_offset, neuron_count, synapse_offset, synapse_count); 

	printf("=====================Arrange Neuron============================\n");
	arrangeNeuron(net, type_offset, neuron_offset, neuron_count); 
	printf("===================Arrange Connection==========================\n");
	for (auto d=_min_delay; d<_max_delay+1; d++) {
		arrangeLocal(net, type_offset, neuron_offset, synapse_offset, neuron_count, synapse_count, n2s_count, d);
		arrangeCross(net, type_offset, synapse_count, n2s_count, cross_idx, node_n_offset, d);
	}

	for (auto node = _neuron_nums.begin(); node != _neuron_nums.end(); node++) {
		for (auto t = node->second.begin(); t != node->second.end(); t++) {
			assert(neuron_count[node->first][t->first] == t->second);
		}
	}

	for (int  node_t = 0; node_t < _node_num; node_t++) {
		assert(cross_idx[node_t] == _crossnodeNeuronsSend[node_t].size());
		assert(node_n_offset[node_t] == _crossnodeNeuronsRecv[node_t].size());
		for (auto i = _synapse_nums[node_t].begin(); i != _synapse_nums[node_t].end(); i++) {
			assert(synapse_count[node_t][i->first] == i->second);
		}
	}

	print_mem("Finish build");

	return net;
}
