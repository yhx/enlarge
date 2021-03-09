
#include "../utils/TypeFunc.h"
#include "../utils/proc_info.h"
#include "../utils/utils.h"
#include "../../include/Synapses.h"
#include "Network.h"
#include "Network.h"

void Network::update_status_nodes()
{
	update_status();
	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_iter->first;
		for (size_t i=0; i<t_iter->second.size(); i++) {
			ID id(t, 0, i);
			unsigned n_idx = _nid2node[id];
			_neuron_nums[n_idx][t] += 1;
			for (auto iter = n2s_conn[id].begin(); iter != n2s_conn[id].end(); iter++) {
				for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
					unsigned int s_idx = _sid2node[*siter];
					_synapse_nums[s_idx][siter->type()] += 1;
				}
			}
		}
	}
}

int Network::arrangeNet(DistriNetwork *net) 
{
	map<unsigned int, map<Type, size_t>> type_offset;
	map<unsigned int, map<Type, size_t>> neuron_offset;
	map<unsigned int, map<Type, size_t>> synapse_offset;
	map<unsigned int, map<Type, size_t>> neuron_count;
	map<unsigned int, map<Type, size_t>> synapse_count;
	map<unsigned int, map<Type, size_t>> n2s_count;

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
	}

	// TODO change d to outer loop
	for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		Type t = t_iter->first;
		for (size_t i=0; i<t_iter->second.size(); i++) {
			ID id(t, 0, i);
			unsigned n_node = _nid2node[id];
			n2s_count[n_node][t] = 0;
			_neurons[t]->packup(net[n_node]._network->ppNeurons[type_offset[n_node][t]], neuron_count[n_node][t], i);
			for (auto iter = n2s_conn[id].begin(); iter != n2s_conn[id].end(); iter++) {
				unsigned int d = iter->first;
				size_t n_offset = (n_num[n_node]+_crossnodeNeuronsRecv[n_node].size())*(d-_min_delay)+neuron_offset[n_node][t]+neuron_count[n_node][t];
				for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
					unsigned int s_node = _sid2node[*siter];
					if (s_node == n_node) {
						Type s_t = iter->second.type();
						unsigned int s_idx = type_offset[n_node][s_t];
						_synapses[t]->packup(net[n_node]._network->ppSynapses[s_idx], synapse_count[n_node][s_t], iter->second.id());
						Connection * c = net[n_node]._network->ppConnections[s_idx];
						c->pDelayStart[n_offset] = n2s_count[n_node][t];
						c->pSidMap[synapse_count[n_node][s_t]] = synapse_count[n_node][s_t];
					}
				}

			}
			neuron_count[n_node][t]++;
		}
	}
	return 0;
}

GNetwork* Network::arrangeData(int nodeIdx, const SimInfo &info) {
	int nTypeNum = _globalNTypeNum[nodeIdx].size();
	int sTypeNum = _globalSTypeNum[nodeIdx].size();

	GNetwork * net = allocGNetwork(nTypeNum, sTypeNum);

	int maxDelaySteps = _max_delay;
	int minDelaySteps = _min_delay;
	int delayLength = maxDelaySteps - minDelaySteps + 1;

	int pindex = 0;
	for (auto tIter = _globalNTypeNum[nodeIdx].begin(); tIter != _globalNTypeNum[nodeIdx].end(); tIter++) {
		Type type = tIter->first;
		net->pNTypes[index] = tIter->first;
		net->ppNeurons[index] = allocType[type](tIter->second);
		assert(net->ppNeurons[index] != NULL);

	        size_t idx = 0;
		for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end();  pIter++) {
			Population * p = *pIter;
#if 1
			for (auto nIter = p->_items.begin(); nIter != p->_items.end(); nIter++) {
				Neuron *n = *nIter;
				if (n->getNode() == nodeIdx && n->getType() == type) {
					size_t copied = n->hardCopy(net->ppNeurons[index], idx, net->pNeuronNums[index], info);
					idx += copied;
				}
				// if (info.save_mem) {
				// 	delete n;
				// 	*nIter = NULL;
				// }
			}
#else
			int node = p->getNode();

			if (node == nodeIdx && p->getType() == type) {
				size_t copied = p->hardCopy(net->ppNeurons[index], idx, net->pNeuronNums[index], info);
				idx += copied;
			}
#endif
		}

		if (idx != tIter->second) {
			printf("Not match: %lu/%llu \n", idx, tIter->second);
		}
		assert(idx == tIter->second);
		net->pNeuronNums[index+1] = idx + net->pNeuronNums[index];
		index++;
	}
	assert(index == nTypeNum);

	index = 0;
	for (auto tIter = _globalSTypeNum[nodeIdx].begin(); tIter != _globalSTypeNum[nodeIdx].end(); tIter++) {
		Type type = tIter->first;
		net->pSTypes[index] = type;
		net->ppSynapses[index] = allocType[type](tIter->second);
		assert(net->ppSynapses[index] != NULL);

		size_t idx = 0;
		for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
			Population *pop = *pIter;
#if 0
			if (pop->getNode() != nodeIdx)
				continue;

			for (size_t nidx=0; nidx<pop->getNum(); nidx++) {
				Neuron *n = pop->locate(nidx);

				vector<Synapse *> &s_vec = pop->locate(nidx)->getSynapses();
#endif
#if 1
			for (auto nIter = pop->_items.begin(); nIter != pop->_items.end(); nIter++) {
				Neuron *n = *nIter;
				if (n->getNode() != nodeIdx)
					continue;

				vector<Synapse *> &s_vec = n->getSynapses();
#endif
				for (int delay_t=0; delay_t<delayLength; delay_t++) {
					for (auto siter = s_vec.begin(); siter != s_vec.end(); siter++) {
						if ((*siter)->getDelaySteps(info.dt) == delay_t + minDelaySteps) {
							if ((*siter)->getType() == type && (*siter)->getNode() == nodeIdx) {
								if (idx >= tIter->second) {
									printf("Overflow: %lu/%llu \n", idx, tIter->second);
								}
								assert(idx < tIter->second);
								int copied = (*siter)->hardCopy(net->ppSynapses[index], idx, net->pSynapseNums[index], info);
								idx += copied;
								// if (info.save_mem) {
								// 	Synapse *s_t = *siter;
								// 	delete s_t;
								// 	*siter = NULL;
								// }
							}
						}
					}
				}
			}
		}

		for (auto niter = _crossnodeNeuronsRecv[nodeIdx].begin(); niter != _crossnodeNeuronsRecv[nodeIdx].end(); niter++) {
			const vector<Synapse *> &s_vec = (*niter)->getSynapses();
			for (int delay_t=0; delay_t<delayLength; delay_t++) {
				for (auto iter = s_vec.begin(); iter != s_vec.end(); iter++) {
					if (((*iter)->getNode() == nodeIdx) && ((*iter)->getDelaySteps(info.dt) == delay_t + minDelaySteps) && ((*iter)->getType() == type)) {
						if (idx >= tIter->second) {
							printf("CrossNode overflow: %lu/%llu \n", idx, tIter->second);
						}
						assert(idx < tIter->second);
						int copied = (*iter)->hardCopy(net->ppSynapses[index], idx, net->pSynapseNums[index], info);
						idx += copied;
					}
				}
			}
		}

		assert(idx == tIter->second); 
		if (idx != tIter->second) {
			printf("CrossNode not match: %lu/%llu \n", idx, tIter->second);
		}
		net->pSynapseNums[index+1] = idx + net->pSynapseNums[index];
		index++;
	}
	assert(index == sTypeNum);

	int nodeNNum = net->pNeuronNums[net->nTypeNum];

	for (auto iter = _crossnodeNeuronsRecv[nodeIdx].begin(); iter !=  _crossnodeNeuronsRecv[nodeIdx].end(); iter++) {
		int size = _crossnodeNeuron2idx[nodeIdx].size();
		_crossnodeNeuron2idx[nodeIdx][*iter] = nodeNNum + size;
	}

	return net;
}

Connection* Network::arrangeConnect(size_t nNum, size_t sNum, int node_idx, const SimInfo &info)
{
	int maxDelaySteps = static_cast<int>(round(_maxDelay/info.dt));
	int minDelaySteps = static_cast<int>(round(_minDelay/info.dt));
	Connection *connection = allocConnection(nNum, sNum, maxDelaySteps, minDelaySteps);
	assert(connection != NULL);

	int delayLength = maxDelaySteps - minDelaySteps + 1;
	size_t synapseIdx = 0;
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
		Population * p = *pIter;
#if 0
		if (p->getNode() != node_idx) 
			continue;
#endif

		for (size_t i=0; i<p->getNum(); i++) {
#if 1
			Neuron *n = p->locate(i);
			if (n->getNode() != node_idx) {
				continue;
			}
#endif
			ID nid = p->locate(i)->getID();
			const vector<Synapse *> &s_vec = p->locate(i)->getSynapses();
			for (int delay_t=0; delay_t<delayLength; delay_t++) {
				connection->pDelayStart[delay_t + delayLength*nid] = synapseIdx;

				for (auto iter = s_vec.begin(); iter != s_vec.end(); iter++) {
					if (((*iter)->getNode() == node_idx) && ((*iter)->getDelaySteps(info.dt) == delay_t + minDelaySteps)) {
						// int sid = (*iter)->getID();
						assert(synapseIdx < sNum);
						synapseIdx++;
					}
				}
				connection->pDelayNum[delay_t + delayLength*nid] = synapseIdx - connection->pDelayStart[delay_t + delayLength*nid];
			}
		}
	}

	for (auto niter = _crossnodeNeuronsRecv[node_idx].begin(); niter != _crossnodeNeuronsRecv[node_idx].end(); niter++) {
		int nid = _crossnodeNeuron2idx[node_idx][*niter];
		const vector<Synapse *> &s_vec = (*niter)->getSynapses();
		for (int delay_t=0; delay_t<delayLength; delay_t++) {
			connection->pDelayStart[delay_t + delayLength*nid] = synapseIdx;

			for (auto iter = s_vec.begin(); iter != s_vec.end(); iter++) {
				if (((*iter)->getNode() == node_idx) && ((*iter)->getDelaySteps(info.dt) == delay_t + minDelaySteps)) {
					// int sid = (*iter)->getID();
					assert(synapseIdx < sNum);
					synapseIdx++;
				}
			}
			connection->pDelayNum[delay_t + delayLength*nid] = synapseIdx - connection->pDelayStart[delay_t + delayLength*nid];
		}
	}

	return connection;
}

// Should finish data arrange of all nodes first.
CrossNodeMap* Network::arrangeCrossNodeMap(size_t n_num, int node_idx, int node_num)
{
	CrossNodeMap* crossMap = (CrossNodeMap*)malloc(sizeof(CrossNodeMap));
	assert(crossMap != NULL);
	crossMap->_num = n_num;

	crossMap->_idx2index = (int*)malloc(sizeof(int)*n_num);
	assert(crossMap->_idx2index != NULL);
	std::fill(crossMap->_idx2index, crossMap->_idx2index + n_num, -1);

	if (_crossnodeNeuronsSend[node_idx].size() > 0) {
		crossMap->_crossnodeIndex2idx = (int*)malloc(sizeof(int) * node_num * _crossnodeNeuronsSend[node_idx].size());
	} else {
		crossMap->_crossnodeIndex2idx = NULL;
	}
	crossMap->_crossSize = node_num * _crossnodeNeuronsSend[node_idx].size();



	int index = 0;
	for (auto iter = _crossnodeNeuronsSend[node_idx].begin(); iter != _crossnodeNeuronsSend[node_idx].end(); iter++) {
		int nidx = (*iter)->getID();
		crossMap->_idx2index[nidx] = index;
		for (int t=0; t<node_num; t++) {
			if (_crossnodeNeuronsRecv[t].find(*iter) != _crossnodeNeuronsRecv[t].end()) {
				assert(crossMap->_crossnodeIndex2idx != NULL);
				crossMap->_crossnodeIndex2idx[index*node_num + t] = _crossnodeNeuron2idx[t][*iter];
			} else {
				assert(crossMap->_crossnodeIndex2idx != NULL);
				crossMap->_crossnodeIndex2idx[index*node_num + t] = -1;
			}
		}
		index++;
	}

	return crossMap;
}

DistriNetwork* Network::buildNetworks(const SimInfo &info, bool auto_splited)
{
	print_mem("Before build");
	printf("===================Update Status================================\n");
	update_status_nodes();

	assert(_node_num >= 1);
	DistriNetwork * net = initDistriNet(_node_num, _dt);

	printf("=====================Split Network=============================\n");
	if (auto_splited && _node_num > 1) {
		splitNetwork();
	}

	printf("=====================Arrange Connect===========================\n");
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
