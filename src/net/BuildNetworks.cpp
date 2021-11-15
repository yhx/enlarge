

#include <iostream>

#include "../base/TypeFunc.h"
#include "../utils/proc_info.h"
#include "../utils/utils.h"
#include "../../include/Synapses.h"
#include "Network.h"
#include "Network.h"

using std::cout;
using std::endl;

void Network::update_status_splited()
{
	_neuron_nums.clear();
	_synapse_nums.clear();
	_crossnodeNeuronsRecv.clear();
	_crossnodeNeuronsSend.clear();
	_crossnodeNeuronsSend.resize(_node_num);
	_crossnodeNeuronsRecv.resize(_node_num);
	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_iter->first;  // neuron type
		for (size_t i=0; i<t_iter->second->size(); i++) {
			bool cross_node = false;  // 是否有跨节点连接，默认为false
			ID id(t, 0, i);
			unsigned n_node = _idx2node[t][i];
			_neuron_nums[n_node][t] += 1;
			// 遍历神经元连接的所有突触
			for (auto iter = _conn_n2s[t][i].begin(); iter != _conn_n2s[t][i].end(); iter++) {
				for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
					unsigned int s_node = _idx2node[siter->type()][siter->id()];  // 突触节点
					_synapse_nums[s_node][siter->type()] += 1;
					if (n_node != s_node) {
						cross_node = true;
						_crossnodeNeuronsRecv[s_node].insert(id);  // 放到跨界点的buffer中去
					}
				}
			}
			if (cross_node) {
				_crossnodeNeuronsSend[n_node].insert(id);
			}
		}
	}

	/**
	 * Deal with poisson synapse. 
	 * Add synapse total number in corresponding node.
	 **/
	if (_synapses.find(Poisson) != _synapses.end()) {
		for (size_t i = 0; i < _synapses[Poisson]->size(); i++) {
			unsigned int s_node = _idx2node[Poisson][i];  // i-th突触所在节点
			_synapse_nums[s_node][Poisson] += 1;
		}
	}

	_buffer_offsets.clear();
	for (int node=0; node<_node_num; node++) {
		size_t count = 0;
		for (auto iter=_neuron_nums[node].begin(); iter!=_neuron_nums[node].end(); iter++) {
			_buffer_offsets[node][iter->first] = count;
			count += iter->second * _neurons[iter->first]->buffer_size();
		}
		_buffer_offsets[node][TYPESIZE] = count;
	}

	cout << "Neuron:" << endl;
	for (auto iter=_neuron_nums.begin(); iter!=_neuron_nums.end(); iter++) {
		cout << "Node " << iter->first << ":" << endl;
		for (auto j=iter->second.begin(); j!=iter->second.end(); j++) {
			cout << "Type " << j->first << " num: " << j->second << endl;
			assert(j->second <= INT_MAX);
		}
	}

	cout << "Synapse:" << endl;
	for (auto iter=_synapse_nums.begin(); iter!=_synapse_nums.end(); iter++) {
		cout << "Node " << iter->first << ":" << endl;
		for (auto j=iter->second.begin(); j!=iter->second.end(); j++) {
			cout << "Type " << j->first << " num: " << j->second << endl;
			assert(j->second <= INT_MAX);
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
			net[i]._network->bufferOffsets[t_offset] = _buffer_offsets[i][n->first];
			assert(net[i]._network->ppNeurons[t_offset] != NULL);
			neuron_count[i][n->first] = 0;
			type_offset[i][n->first] = t_offset;
			neuron_offset[i][n->first] = n_offset;
			t_offset++;
			n_offset += n->second;
		}
		net[i]._network->pNeuronNums[t_offset] = n_offset;
		net[i]._network->bufferOffsets[t_offset] = _buffer_offsets[i][TYPESIZE];
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
			unsigned int n_node = _idx2node[t][i];
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
			unsigned int n_node = _idx2node[t][i];
		    size_t n_num = net[n_node]._network->pNeuronNums[_neuron_nums[n_node].size()];
			size_t n_offset = (n_num+_crossnodeNeuronsRecv[n_node].size())*(delay-_min_delay)+_id2node_idx[id];
			for (auto siter = _conn_n2s[t][i][delay].begin(); siter != _conn_n2s[t][i][delay].end(); siter++) {
				unsigned int s_node = _idx2node[siter->type()][siter->id()];
				if (s_node == n_node) {
					Type s_t = siter->type();
					unsigned int s_idx = type_offset[s_node][s_t];
					_id2node_idx[*siter] = synapse_count[s_node][s_t];
					_synapses[s_t]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][s_t], siter->id());
					Connection * c = net[s_node]._network->ppConnections[s_idx];
					c->pDelayStart[n_offset] = n2s_count[s_node][s_t];
					c->pSidMap[synapse_count[s_node][s_t]] = synapse_count[s_node][s_t];
					ID target = _conn_s2n[siter->type()][siter->id()];
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

	/**
	 * Deal with poisson synapse.
	 */
	if (_conn_s2n.find(Poisson) != _conn_s2n.end()) {
		if (_conn_sd2n[Poisson].find(delay) != _conn_sd2n[Poisson].end()) {  // if has specific delay
			for (auto pr = _conn_sd2n[Poisson][delay].begin(); pr < _conn_sd2n[Poisson][delay].end(); pr++) {
				// pr is pair<src_idx, dst_ID>
			// for (int i = 0; i < _conn_s2n[Poisson].size(); ++i) {
				int i = pr->first;  // src node index
				unsigned int s_node = _idx2node[Poisson][i];
				ID s_id(Poisson, 0, i);  // 当前突触的id
				unsigned int s_idx = type_offset[s_node][Poisson];
				_id2node_idx[s_id] = synapse_count[s_node][Poisson];  // 当前类型突触的开始编号
				_synapses[Poisson]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][Poisson], i);
				Connection * c = net[s_node]._network->ppConnections[s_idx];
				c->pSidMap[synapse_count[s_node][Poisson]] = synapse_count[s_node][Poisson];
				// ID target = _conn_s2n[Poisson][i];
				ID target = pr->second;
				c->dst[synapse_count[s_node][Poisson]] = _buffer_offsets[s_node][target.type()] + target.offset() * _neuron_nums[s_node][target.type()] + _id2node_idx[target.mask_offset()];
				synapse_count[s_node][Poisson]++;
			// }
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
				assert(n_node == _idx2node[iter->type()][iter->id()]);
				assert(cross_idx[n_node] < _crossnodeNeuronsSend[n_node].size());
				map->_idx2index[_id2node_idx[*iter]] = cross_idx[n_node];
				cross_idx[n_node]++;
			}
		}
	}

	for (int s_node = 0; s_node < _node_num; s_node++) {
		for (auto iter = _crossnodeNeuronsRecv[s_node].begin(); iter != _crossnodeNeuronsRecv[s_node].end(); iter++) {
			int n_node = _idx2node[iter->type()][iter->id()];
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

			for (auto siter = _conn_n2s[iter->type()][iter->id()][delay].begin(); siter != _conn_n2s[iter->type()][iter->id()][delay].end(); siter++) {
				if (s_node == _idx2node[siter->type()][siter->id()]) {
					Type s_t = siter->type();
					size_t s_idx = type_offset[s_node][s_t];
					_synapses[s_t]->packup(net[s_node]._network->ppSynapses[s_idx], synapse_count[s_node][s_t], siter->id());
					Connection * c = net[s_node]._network->ppConnections[s_idx];
					c->pDelayStart[n_offset] = n2s_count[s_node][s_t];
					c->pSidMap[synapse_count[s_node][s_t]] = synapse_count[s_node][s_t];
					ID target = _conn_s2n[siter->type()][siter->id()];
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

DistriNetwork* Network::buildNetworks(const SimInfo &info, SplitType split, const char *name, const AlgoPara *para, bool auto_splited)
{
	print_mem("Before build");
	printf("===Update Status\n");
	update_status();

	assert(_node_num >= 1);
	DistriNetwork * net = initDistriNet(_node_num, _dt);

	printf("===Split Network\n");
	if (auto_splited) {
		splitNetwork(split, name, para);
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


	printf("===Arrange Network\n");
	arrangeNet(net, type_offset, neuron_offset, neuron_count, synapse_offset, synapse_count); 
	print_mem("after arrange network");

	printf("===Arrange Neuron\n");
	arrangeNeuron(net, type_offset, neuron_offset, neuron_count); 
	print_mem("after arrange neuron");
	printf("===Arrange Connection\n");
	for (auto d=_min_delay; d<_max_delay+1; d++) {
		arrangeLocal(net, type_offset, neuron_offset, synapse_offset, neuron_count, synapse_count, n2s_count, d);
		print_mem("after arrange local");
		arrangeCross(net, type_offset, synapse_count, n2s_count, cross_idx, node_n_offset, d);
		print_mem("after arrange cross");
	}
	print_mem("after arrange connection");

	for (auto node = _neuron_nums.begin(); node != _neuron_nums.end(); node++) {
		for (auto t = node->second.begin(); t != node->second.end(); t++) {
			assert(neuron_count[node->first][t->first] == t->second);
		}
	}

	for (int node_t = 0; node_t < _node_num; node_t++) {
		assert(cross_idx[node_t] == _crossnodeNeuronsSend[node_t].size());
		assert(node_n_offset[node_t] == _crossnodeNeuronsRecv[node_t].size());
		for (auto i = _synapse_nums[node_t].begin(); i != _synapse_nums[node_t].end(); i++) {
			assert(synapse_count[node_t][i->first] == i->second);
		}
	}

	print_mem("Finish build");

	return net;
}
