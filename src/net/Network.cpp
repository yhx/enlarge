/* This program is writen by qp09.
 * usually just for fun.
 * Sat October 24 2015
 */

#include <math.h>
#include <limits.h>
#include <chrono>
#include <sys/sysinfo.h>

#include "../utils/TypeFunc.h"
#include "../utils/proc_info.h"
#include "../utils/utils.h"
#include "Network.h"

#define SYN_BASE 0
#define NEU_BASE 1
#define ROUND_ROBIN 2
#define BALANCED 3
#define SYN_POP 100

#define SPLIT ROUND_ROBIN 

using namespace std::chrono;

Network::Network(int node_num)
{
	_max_delay = 0;
	_min_delay = INT_MAX;
	// maxDelaySteps = 0;
	// minDelaySteps = 1e7;
	// _maxFireRate = 0.0;

	_node_num = node_num;
	//n2sNetwork.clear();
	
	_crossnodeNeuronsSend.resize(node_num);
	_crossnodeNeuronsRecv.resize(node_num);
	_crossnodeNeuron2idx.resize(node_num);

	_globalNTypeNum.resize(node_num);
	_globalSTypeNum.resize(node_num);
}

Network::~Network()
{
	del_content_vec(_populations);
	del_content_map(_neurons);
	del_content_map(_synapses);

	print_mem("Free Network 1");
	// _neuronNums.clear();
	// _connectNums.clear();
	// _synapseNums.clear();
	// _nTypes.clear();
	// _sTypes.clear();

	_crossnodeNeuronsSend.clear();
	_crossnodeNeuronsRecv.clear();
	_crossnodeNeuron2idx.clear();

	_globalNTypeNum.clear();
	_globalSTypeNum.clear();

	print_mem("Free Network 2");
}

int Network::setNodeNum(int node_num)
{
	_node_num = node_num;

	_crossnodeNeuronsSend.clear();
	_crossnodeNeuronsRecv.clear();
	_crossnodeNeuron2idx.clear();

	_globalNTypeNum.clear();
	_globalSTypeNum.clear();

	_crossnodeNeuronsSend.resize(node_num);
	_crossnodeNeuronsRecv.resize(node_num);
	_crossnodeNeuron2idx.resize(node_num);

	_globalNTypeNum.resize(node_num);
	_globalSTypeNum.resize(node_num);

	return _node_num;
}

int Network::connect(ID src, ID dst, ID syn, unsigned int delay)
{
	n2s_conn[src][delay].push_back(syn);
	s2n_conn[syn] = dst;
	return 1;
}

int Network::connect(Population *p_src, Population *p_dst, real weight, real delay, real tau, SpikeType type) {
	size_t src_size = p_src->size();
	size_t dst_size = p_dst->size();
	size_t size = src_size * dst_size; 

	Type type = Static;
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new Synapse(weight, delay, tau, _dt, size);
	} else {
		offset = _synapses.size();
		Synapse t(weight, delay, tau, _dt, size);
		_synapses[type]->append(&t, size);
	}

	int count = 0;
	for (size_t s=0; s<src_size; s++) {
		for (size_t d=0; d<dst_size; d++) {
			size_t s_offset = offset + s*dst_size + d;
			connect(ID(p_src->type(), 0, p_src->offset()+s), 
					ID(p_dst->type(), sp, p_dst->offset()+d),
				   ID(type, 0, s_offset),
				   _synapses[type]->get_delay()[s_offset]);
			count++;
		}
	}

	return count;
}

int Network::connect(Population *p_src, Population *p_dst, real *weight, real *delay, real *tau, SpikeType *type, size_t size) {
	size_t dst_size = p_dst->get_size();
	assert(size == (p_src->size() * dst_size)); 

	Type type = Static;
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new Synapse(weight, delay, tau, _dt, size);
	} else {
		offset = _synapses.size();
		Synapse t(weight, delay, tau, _dt, size);
		_synapses[type]->append(&t, size);
	}

	int count = 0;
	for (size_t s=0; s<src_size; s++) {
		for (size_t d=0; d<dst_size; d++) {
			size_t s_offset = offset + s*dst_size + d;
			connect(ID(p_src->type(), 0, p_src->offset()+s), 
					ID(p_dst->type(), sp, p_dst->offset()+d),
				   ID(type, 0, s_offset),
				   _synapses[type]->get_delay()[s_offset]);
			count++;
		}
	}

	return count;
}

int connect(Population *p_src, size_t src, Population *p_dst, size_t dst, real weight, real delay, real tau, SpikeType=Exc) 
{
	Type type = Static;
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new Synapse(weight, delay, tau, _dt);
	} else {
		offset = _synapses.size();
		Synapse t(weight, delay, tau, _dt);
		_synapses[type]->append(&t, size);
	}

	size_t s_offset = offset + s*dst_size + d;
	connect(ID(p_src->type(), 0, p_src->offset()), 
			ID(p_dst->type(), sp, p_dst->offset()),
			ID(type, 0, s_offset),
			_synapses[type]->get_delay()[s_offset]);

	return count;
}

// int Network::connectOne2One(Population *pSrc, Population *pDst, real *weight, real *delay, SpikeType *type, size_t size) {
// 	assert(size == pSrc->getNum());
// 	assert(size == pDst->getNum()); 
// 
// 	if (find(_pPopulations.begin(), _pPopulations.end(), pSrc) == _pPopulations.end()) {
// 		_pPopulations.push_back(pSrc);
// 		_populationNum++;
// 		//neuronNum += pSrc->getNum();
// 		addNeuronNum(pSrc->getType(), pSrc->getNum());
// 	}
// 	if (find(_pPopulations.begin(), _pPopulations.end(), pDst) == _pPopulations.end()) {
// 		_pPopulations.push_back(pDst);
// 		_populationNum++;
// 		//neuronNum += pDst->getNum();
// 		addNeuronNum(pDst->getType(), pDst->getNum());
// 	}
// 
// 	int count = 0;
// 	for (size_t i=0; i<size; i++) {
// 		if (type == NULL) {
// 			connect(pSrc->locate(i), pDst->locate(i), weight[i], delay[i], Excitatory, 0.0, false);
// 		} else {
// 			connect(pSrc->locate(i), pDst->locate(i), weight[i], delay[i], type[i], 0.0, false);
// 		}
// 		count++;
// 	}
// 
// 	return count;
// }
// 
// int Network::connectConv(Population *pSrc, Population *pDst, real *weight, real *delay, SpikeType *type, size_t height, size_t width, size_t k_height, size_t k_width) {
// 	assert(pSrc->getNum() == height * width); 
// 	assert(pDst->getNum() == height * width); 
// 
//         size_t count = 0;
// 	for (size_t h = 0; h < height; h++) {
// 		for (size_t w = 0; w < width; w++) {
// 			for (size_t i = 0; i< k_height; i++) {
// 				for (size_t j = 0; j < k_width; j++) {
// 					size_t idx_h = h + i - (k_height - 1)/2;
// 					size_t idx_w = w + j - (k_width - 1)/2;
// 
// 					if (idx_h >= 0 && idx_h < height && idx_w >= 0 && idx_w < width) {
// 						count++;
// 						if (type == NULL) {
// 							connect(pSrc->locate(idx_h * width + idx_w), pDst->locate(h * width + w), weight[i*k_width + j], delay[i*k_width + j], Excitatory, 0.0, false);
// 
// 						} else {
// 							connect(pSrc->locate(idx_h * width + idx_w), pDst->locate(h * width + w), weight[i*k_width + j], delay[i*k_width + j], type[i*k_width + j], 0.0, false);
// 						}
// 					}
// 				}
// 			}
// 		}	
// 	}
// 
// 	return count;
// }
// 
// int Network::connectPooling(Population *pSrc, Population *pDst, real delay, size_t height, size_t width, size_t p_height, size_t p_width)
// {
// 	assert(pDst->getNum() == pSrc->getNum() / p_height / p_width); 
// 
// 	//size_t d_height = height/p_height;
// 	size_t d_width = width/p_width;
// 
// 	size_t count = 0;
// 	for (size_t h = 0; h < height; h++) {
// 		for (size_t w = 0; w < width; w++) {
// 			size_t d_h = h/p_height;
// 			size_t d_w = w/p_width;
// 			size_t d_h_ = h % p_height;
// 			size_t d_w_ = w % p_width;
// 			size_t idx = d_h_ * p_width + d_w_;
// 
// 			count++;
// 			connect(pSrc->locate(h * width + w), pDst->locate(d_h*d_width + d_w), (real)(1 << idx), delay, Excitatory, 0.0, false);
// 		}	
// 	}
// 
// 	return count;
// }



int Network::reset(const SimInfo &info)
{
	_crossnodeNeuronsSend.clear();
	_crossnodeNeuronsRecv.clear();
	_crossnodeNeuron2idx.clear();

	_globalNTypeNum.clear();
	_globalSTypeNum.clear();
	
	return 0;
}
 

void Network::logMap() {
}

CrossThreadData* Network::arrangeCrossThreadData(int node_num)
{
	CrossThreadData * cross_data = (CrossThreadData*)malloc(sizeof(CrossThreadData) * node_num * node_num);
	assert(cross_data != NULL);

	for (int i=0; i<_node_num; i++) {
		for (int j=0; j<_node_num; j++) {
			// i->j 
			int i2j = j * _node_num + i;
			cross_data[i2j]._firedNNum = 0;

			int count = 0;
			for (auto iter = _crossnodeNeuronsSend[i].begin(); iter != _crossnodeNeuronsSend[i].end(); iter++) {
				if (_crossnodeNeuronsRecv[j].find(*iter) != _crossnodeNeuronsRecv[j].end()) {
					count++;
				}
			}
			cross_data[i2j]._maxNNum = count;
			cross_data[i2j]._firedNIdxs = (int*)malloc(sizeof(int)*count);
			assert(cross_data[i2j]._firedNIdxs != NULL || count == 0);
		}
	}


	for (int i=0; i<_node_num; i++) {
		int idx = i*_node_num + i;
		for (int j=0; j<_node_num; j++) {
			if (j != i) {
				cross_data[idx]._maxNNum += cross_data[i*_node_num+j]._maxNNum;
			}
		}

		cross_data[idx]._firedNIdxs = (int*)malloc(sizeof(int)*cross_data[idx]._maxNNum);
		assert(cross_data[idx]._firedNIdxs != NULL || cross_data[idx]._maxNNum == 0);
	}

	return cross_data;
}

CrossNodeData* Network::arrangeCrossNodeData(int node_num, const SimInfo &info)
{
	CrossNodeData * cross_data = (CrossNodeData*)malloc(sizeof(CrossNodeData) * node_num);
	assert(cross_data != NULL);

	int delay_t = static_cast<int>(round(_minDelay/info.dt));
	for (int i=0; i<node_num; i++) {
		allocParaCND(&(cross_data[i]), node_num, delay_t);
	}

	for (int i=0; i<node_num; i++) {
		cross_data[i]._send_offset[0] = 0;
		for (int j=0; j<node_num; j++) {

			int count = 0;
			for (auto iter = _crossnodeNeuronsSend[i].begin(); iter != _crossnodeNeuronsSend[i].end(); iter++) {
				if (_crossnodeNeuronsRecv[j].find(*iter) != _crossnodeNeuronsRecv[j].end()) {
					count++;
				}
			}
			cross_data[i]._send_offset[j+1] = cross_data[i]._send_offset[j] + count * delay_t;
		}
		// cross_data[i]._send_data = (int*)malloc(sizeof(int)*(cross_data[i]._send_offset[node_num]));
		// assert(cross_data[i]._send_data != NULL || cross_data[i]._send_offset[node_num] == 0);
	}


	for (int i=0; i<node_num; i++) {
		cross_data[i]._recv_offset[0] = 0;
		assert(0 ==  cross_data[i]._send_offset[i+1] - cross_data[i]._send_offset[i]); 
		for (int j=0; j<node_num; j++) {
			int count_t = cross_data[j]._send_offset[i+1] - cross_data[j]._send_offset[i]; 
			cross_data[i]._recv_offset[j+1] = cross_data[i]._recv_offset[j] + count_t;
		}
		// cross_data[i]._recv_data = (int*)malloc(sizeof(int)*(cross_data[i]._recv_offset[node_num]));
		// assert(cross_data[i]._recv_data != NULL || cross_data[i]._recv_offset[node_num] == 0);
	}

	return cross_data;
}

void printTypeNum(vector<map<Type, unsigned long long>> typeNum, const char *name) {
	printf("The %s number: ", name);
	for (size_t i=0; i<typeNum.size(); i++) {
		unsigned long long  total = 0;
		for (auto tIter = typeNum[i].begin(); tIter != typeNum[i].end(); tIter++) {
			Type type = tIter->first;
			unsigned long long num = tIter->second;
			printf("%d_%llu@%lu\t", type, num, i);
			total += num;
		}
		assert(total < INT_MAX);
	}
	printf("\n");
}

void Network::countTypeNum() 
{
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end();  pIter++) {
		Population * p = *pIter;

#if 1
		for (auto nIter = p->_items.begin(); nIter != p->_items.end(); nIter++) {
			Neuron *n = *nIter;
			Type type = n->getType();
			int node = n->getNode();
			if (_globalNTypeNum[node].find(type) == _globalNTypeNum[node].end()) {
				_globalNTypeNum[node][type] = 1;
			} else {
				_globalNTypeNum[node][type] += 1;
			}
		}
#else
		Type type = p->getType();
		int node = p->getNode();
		if (_globalNTypeNum[node].find(type) == _globalNTypeNum[node].end()) {
			_globalNTypeNum[node][type] = p->getNum();
		} else {
			_globalNTypeNum[node][type] += p->getNum();
		}
#endif
	}

// #ifdef PROF // Too costy // Too costy
// 	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end();  pIter++) {
// 		Population *p = *pIter;
// 		for (auto nIter = p->_items.begin(); nIter != p->_items.end(); nIter++) {
// 			Neuron *n = *nIter;
// 			const vector<Synapse *> &s_vec = n->getSynapses();
// 			for (auto sIter = s_vec.begin(); sIter != s_vec.end(); sIter++) {
// 				Synapse *s = *sIter;
// 				if (find(_pSynapses.begin(), _pSynapses.end(), s) == _pSynapses.end()) {
// 					printf("Synapse %d (src %d, dst %d) not in global array\n", s->getID(), n->getID(), s->getDst()->getID());
// 				}
// 			}
// 		}
// 
// 	}
// #endif

#if 0
	for (auto siter = _pSynapses.begin(); siter != _pSynapses.end();  siter++) {
		Synapse * p = *siter;
		Type type = p->getType();
		int node = p->getNode();

		if (_globalSTypeNum[node].find(type) == _globalSTypeNum[node].end()) {
			_globalSTypeNum[node][type] = 1;
		} else {
			_globalSTypeNum[node][type] += 1;
		}
	}
#else
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end();  pIter++) {
		Population *p = *pIter;
		for (auto nIter = p->_items.begin(); nIter != p->_items.end(); nIter++) {
			Neuron *n = *nIter;
			const vector<Synapse *> &s_vec = n->getSynapses();
			for (auto sIter = s_vec.begin(); sIter != s_vec.end(); sIter++) {
				Synapse *s = *sIter;
				Type type = s->getType();
				int node = s->getNode();
				if (_globalSTypeNum[node].find(type) == _globalSTypeNum[node].end()) {
					_globalSTypeNum[node][type] = 1;
				} else {
					_globalSTypeNum[node][type] += 1;
				}
			}
		}

	}
#endif

	printTypeNum(_globalNTypeNum, "neuron");
	printTypeNum(_globalSTypeNum, "synapse");

}

GNetwork* Network::arrangeData(int nodeIdx, const SimInfo &info) {
	int nTypeNum = _globalNTypeNum[nodeIdx].size();
	int sTypeNum = _globalSTypeNum[nodeIdx].size();

	GNetwork * net = allocGNetwork(nTypeNum, sTypeNum);

	int maxDelaySteps = static_cast<int>(round(_maxDelay/info.dt));
	int minDelaySteps = static_cast<int>(round(_minDelay/info.dt));
	int delayLength = maxDelaySteps - minDelaySteps + 1;

	int index = 0;
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

void Network::status()
{
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end();  pIter++) {
		Population *p = *pIter;
		for (auto nIter = p->_items.begin(); nIter != p->_items.end(); nIter++) {
			Neuron *n = *nIter;
			printf("%d ", n->getNode());
		}
		printf("; ");
	}
}

void Network::splitNetwork()
{
	map<Neuron *, vector<Synapse *>> n2sInput;

#if 0
	for (auto sIter = _pSynapses.begin(); sIter != _pSynapses.end(); sIter++) {
		Synapse * p = *sIter;
		n2sInput[p->getDst()].push_back(p);
	}
#else
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end();  pIter++) {
		Population *p = *pIter;
		for (auto nIter = p->_items.begin(); nIter != p->_items.end(); nIter++) {
			Neuron *n = *nIter;
			const vector<Synapse *> &s_vec = n->getSynapses();
			for (auto sIter = s_vec.begin(); sIter != s_vec.end(); sIter++) {
				Synapse *s = *sIter;
				n2sInput[s->getDst()].push_back(s);
			}
		}
	}
#endif

	// Check n2s is right
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
		Population * p = *pIter;
		for (size_t i=0; i<p->getNum(); i++) {
			Neuron * n = p->locate(i);
			auto n2sIter = n2sInput.find(n);
			if (n2sIter != n2sInput.end()) {
				for (auto vIter = n2sIter->second.begin(); vIter != n2sIter->second.end(); vIter++) {
					assert((*vIter)->getDst() == n);
				}
			}
		}
	}
	

#if SPLIT==SYN_BASE
	printf("===========SYN_BASE==========\n");
	int nodeIdx = 0;
	unsigned long long  synapseCount = 0;
	unsigned long long synapsePerNode = _totalSynapseNum/_node_num;
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
		Population * p = *pIter;
		// p->setNode(nodeIdx);
		for (size_t i=0; i<p->getNum(); i++) {
			p->locate(i)->setNode(nodeIdx);
			auto n2sIter = n2sInput.find(p->locate(i));
			if (n2sIter != n2sInput.end()) {
				synapseCount += n2sIter->second.size();
				for (auto vIter = n2sIter->second.begin(); vIter != n2sIter->second.end(); vIter++) {
					(*vIter)->setNode(nodeIdx);
				}
			}
			if (synapseCount >= (nodeIdx+1) * synapsePerNode && nodeIdx < _node_num - 1) {
				nodeIdx++;	
			}
		}
	}
#elif SPLIT==NEU_BASE
	printf("===========NEU_BASE==========\n");
	int nodeIdx = 0;
	unsigned long long neuronCount = 0;
	unsigned long long  neuronPerNode = _totalNeuronNum/_node_num;
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
		Population * p = *pIter;
		// p->setNode(nodeIdx);
		for (size_t i=0; i<p->getNum(); i++) {
			p->locate(i)->setNode(nodeIdx);
			auto n2sIter = n2sInput.find(p->locate(i));
			if (n2sIter != n2sInput.end()) {
				// synapseCount += n2sIter->second.size();
				for (auto vIter = n2sIter->second.begin(); vIter != n2sIter->second.end(); vIter++) {
					(*vIter)->setNode(nodeIdx);
				}
			}
			if (neuronCount >= (nodeIdx+1) * neuronPerNode && nodeIdx < _node_num - 1) {
				nodeIdx++;
			}
		}
	}
#elif SPLIT==ROUND_ROBIN
	printf("===========ROUND_ROBIN==========\n");
	unsigned long long neuronCount = 0;
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
		Population * p = *pIter;
		// p->setNode(nodeIdx);
		for (size_t i=0; i<p->getNum(); i++) {
			int nodeIdx = neuronCount % _node_num;
			neuronCount++;
			p->locate(i)->setNode(nodeIdx);
			auto n2sIter = n2sInput.find(p->locate(i));
			if (n2sIter != n2sInput.end()) {
				// synapseCount += n2sIter->second.size();
				for (auto vIter = n2sIter->second.begin(); vIter != n2sIter->second.end(); vIter++) {
					(*vIter)->setNode(nodeIdx);
				}
			}
		}
	}
#elif SPLIT==BALANCED
	printf("===========BALANCED==========\n");
#else
	printf("===========Default==========\n");
	int nodeIdx = 0;
	unsigned long long synapseCount = 0;
	unsigned long long synapsePerNode = _totalSynapseNum/_node_num;
	for (auto pIter = _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
		Population * p = *pIter;
		// p->setNode(nodeIdx);
		for (size_t i=0; i<p->getNum(); i++) {
			p->locate(i)->setNode(nodeIdx);
			auto n2sIter = n2sInput.find(p->locate(i));
			if (n2sIter != n2sInput.end()) {
				synapseCount += n2sIter->second.size();
				for (auto vIter = n2sIter->second.begin(); vIter != n2sIter->second.end(); vIter++) {
					(*vIter)->setNode(nodeIdx);
				}
			}

		}
		if (synapseCount >= (nodeIdx+1) * synapsePerNode && nodeIdx < _node_num - 1) {
			nodeIdx++;	
		}
	}
#endif

	n2sInput.clear();

	for (auto pIter= _pPopulations.begin(); pIter != _pPopulations.end(); pIter++) {
		Population * p = *pIter;
		for (size_t i=0; i<p->getNum(); i++) {
			bool crossNoded = false;
			Neuron *n = p->locate(i);
			int n_node = p->locate(i)->getNode();
			const vector<Synapse *> &tmp = p->locate(i)->getSynapses();
			for (auto iter = tmp.begin(); iter != tmp.end(); iter++) {
				int s_node = (*iter)->getNode();
				if (s_node != n_node) {
					crossNoded = true;
					_crossnodeNeuronsRecv[s_node].insert(n);
				}
			}

			if (crossNoded) {
				_crossnodeNeuronsSend[n_node].insert(n);
			}
		}

	}

	return;
}

DistriNetwork* Network::buildNetworks(const SimInfo &info, bool auto_splited)
{
	print_mem("Before build");

	assert(_node_num >= 1);
	DistriNetwork * net = initDistriNet(_node_num, info.dt);

	printf("=====================Split Network=============================\n");
	if (auto_splited && _node_num > 1) {
		splitNetwork();
	}

	printf("=====================Count Type================================\n");
	countTypeNum();

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
