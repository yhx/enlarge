/* This program is writen by qp09.
 * usually just for fun.
 * Sat October 24 2015
 */

#include <math.h>
#include <limits.h>
#include <chrono>
#include <sys/sysinfo.h>

#include "../base/TypeFunc.h"
#include "../utils/proc_info.h"
#include "../utils/utils.h"
#include "../../include/Synapses.h"
#include "Network.h"

using namespace std::chrono;

Network::Network(real dt, int node_num)
{
	_dt = dt;
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

int Network::set_node_num(int node_num)
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

	n2s_conn_rev[dst.mask_offset()][delay].push_back(syn);
	s2n_conn_rev[syn] = src;
	return 1;
}

int Network::connect(Population *p_src, Population *p_dst, real weight, real delay, real tau, SpikeType sp) {
	size_t src_size = p_src->size();
	size_t dst_size = p_dst->size();
	size_t size = src_size * dst_size; 

	Type type = Static;
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new StaticSynapse(weight, delay, tau, _dt, size);
	} else {
		offset = _synapses.size();
		StaticSynapse t(weight, delay, tau, _dt, size);
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

int Network::connect(Population *p_src, Population *p_dst, real *weight, real *delay, real *tau, SpikeType *sp, size_t size) {
	size_t dst_size = p_dst->size();
	size_t src_size = p_src->size();
	assert(size == (src_size * dst_size)); 

	Type type = Static;
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new StaticSynapse(weight, delay, tau, _dt, size);
	} else {
		offset = _synapses.size();
		StaticSynapse t(weight, delay, tau, _dt, size);
		_synapses[type]->append(&t, size);
	}

	int count = 0;
	for (size_t s=0; s<src_size; s++) {
		for (size_t d=0; d<dst_size; d++) {
			size_t s_offset = offset + s*dst_size + d;
			connect(ID(p_src->type(), 0, p_src->offset()+s), 
					ID(p_dst->type(), sp[s*dst_size+d], p_dst->offset()+d),
				   ID(type, 0, s_offset),
				   _synapses[type]->get_delay()[s_offset]);
			count++;
		}
	}

	return count;
}

int Network::connect(Population *p_src, size_t src, Population *p_dst, size_t dst, real weight, real delay, real tau, SpikeType sp) 
{
	Type type = Static;
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new StaticSynapse(weight, delay, tau, _dt);
	} else {
		offset = _synapses.size();
		StaticSynapse t(weight, delay, tau, _dt);
		_synapses[type]->append(&t);
	}

	size_t s_offset = offset;
	connect(ID(p_src->type(), 0, p_src->offset()+src), 
			ID(p_dst->type(), sp, p_dst->offset()+dst),
			ID(type, 0, s_offset),
			_synapses[type]->get_delay()[s_offset]);

	return 1;
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

// CrossThreadData* Network::arrangeCrossThreadData(int node_num)
// {
// 	CrossThreadData * cross_data = (CrossThreadData*)malloc(sizeof(CrossThreadData) * node_num * node_num);
// 	assert(cross_data != NULL);
// 
// 	for (unsigned int i=0; i<_node_num; i++) {
// 		for (unsigned int j=0; j<_node_num; j++) {
// 			// i->j 
// 			int i2j = j * _node_num + i;
// 			cross_data[i2j]._firedNNum = 0;
// 
// 			int count = 0;
// 			for (auto iter = _crossnodeNeuronsSend[i].begin(); iter != _crossnodeNeuronsSend[i].end(); iter++) {
// 				if (_crossnodeNeuronsRecv[j].find(*iter) != _crossnodeNeuronsRecv[j].end()) {
// 					count++;
// 				}
// 			}
// 			cross_data[i2j]._maxNNum = count;
// 			cross_data[i2j]._firedNIdxs = (int*)malloc(sizeof(int)*count);
// 			assert(cross_data[i2j]._firedNIdxs != NULL || count == 0);
// 		}
// 	}
// 
// 
// 	for (unsigned int i=0; i<_node_num; i++) {
// 		unsigned int idx = i*_node_num + i;
// 		for (unsigned int j=0; j<_node_num; j++) {
// 			if (j != i) {
// 				cross_data[idx]._maxNNum += cross_data[i*_node_num+j]._maxNNum;
// 			}
// 		}
// 
// 		cross_data[idx]._firedNIdxs = (int*)malloc(sizeof(int)*cross_data[idx]._maxNNum);
// 		assert(cross_data[idx]._firedNIdxs != NULL || cross_data[idx]._maxNNum == 0);
// 	}
// 
// 	return cross_data;
// }

CrossNodeData* Network::arrangeCrossNodeData(int node_num, const SimInfo &info)
{
	CrossNodeData * cross_data = (CrossNodeData*)malloc(sizeof(CrossNodeData) * node_num);
	assert(cross_data != NULL);

	int delay_t = _min_delay;
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


void Network::status()
{
}

