#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "MultiNetwork.h"

void MultiNetwork::splitNetwork()
{
	int node_idx = 0;
	int synapse_count = 0;
	int synapse_per_node = _network->totalSynapseNum/_node_num;

	map<Neuron *, vector<Synapse *>> n2s_input;
	for (auto siter = _network->pSynapses.begin(); siter != _network->pSynapses.end(); siter++) {
		Synapse * p = *siter;
		n2s_input[p->getDst()].push_back(p);
	}

	for (auto piter = _network->pPopulations.begin(); piter != _network->pPopulations.end(); piter++) {
		Population * p = *piter;
		p->setNode(node_idx);
		for (int i=0; i<p->getNum(); i++) {
			p->locate(i)->setNode(node_idx);
			auto n2siter = n2s_input.find(p->locate(i));
			if (n2siter != n2s_input.end()) {
				synapse_count += n2siter->second.size();
				for (auto viter = n2siter->second.begin(); viter != n2siter->second.end(); viter++) {
					(*viter)->setNode(node_idx);
				}
			}

		}
		if (synapse_count >= (node_idx+1) * synapse_per_node && node_idx < _node_num - 1) {
			node_idx++;	
		}
	}

	n2s_input.clear();

	for (auto piter= _network->pPopulations.begin(); piter != _network->pPopulations.end(); piter++) {
		Population * p = *piter;
		for (int i=0; i<p->getNum(); i++) {
			bool cross_node = false;
			Neuron *n = p->locate(i);
			int n_node = p->locate(i)->getNode();
			const vector<Synapse *> &tmp = p->locate(i)->getSynapses();
			for (auto iter = tmp.begin(); iter != tmp.end(); iter++) {
				int s_node = (*iter)->getNode();
				if (s_node != n_node) {
					cross_node = true;
					_crossnode_neurons_recv[s_node].insert(n);
				}
			}

			if (cross_node) {
				_crossnode_neurons_send[n_node].insert(n);
			}
		}

	}

	//// Find out cross node neurons
	//for (auto n2siter = n2s_output.begin(); n2siter != n2s_output.end(); n2siter++) {
	//	vector<ID> &p = n2siter->second;
	//	int synapse_num = p.size();
	//	int nnode = _nID2node[n2siter->first];
	//	if (synapse_num > 0) {
	//		bool cross_node = false;
	//		for (vector<ID>::iterator iter = p.begin(); iter != p.end(); iter++) {
	//			int snode = _sID2node[*iter];
	//			if (snode != nnode) {
	//				cross_node = true;
	//				_crossnode_IDs_receive[snode].insert(n2siter->first);
	//			}
	//		}
	//		if (cross_node) {
	//			_crossnode_IDs_send[nnode].insert(n2siter->first);
	//		}
	//	}

	//}

	//Double Check
	//map<ID, set<int> > _crossnode_nID2nodes;
	//for (auto n2siter = _network->n2sNetwork.begin(); n2siter != _network->n2sNetwork.end(); n2siter++) {
	//	vector<ID> &p = n2siter->second;
	//	int synapse_num = p.size();
	//	int baseNode = _nID2node[n2siter->first];
	//	if (synapse_num > 0) {
	//		for (vector<ID>::iterator iter = p.begin(); iter != p.end(); iter++) {
	//			int snode = _sID2node[*iter];
	//			if (snode != baseNode) {
	//				_crossnode_nID2nodes[n2siter->first].insert(snode);
	//			}
	//		}
	//	}

	//}

	//for (auto iter = _crossnode_nID2nodes.begin(); iter != _crossnode_nID2nodes.end(); iter++) {
	//	int nnode = _nID2node[iter->first];
	//	assert(_crossnode_IDs_send[nnode].find(iter->first) != _crossnode_IDs_send[nnode].end());
	//	for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++) {
	//		//int snode = _sID2node[*iter2];
	//		assert(_crossnode_IDs_receive[*iter2].find(iter->first) != _crossnode_IDs_receive[*iter2].end());
	//	}
	//}

	return;
}
