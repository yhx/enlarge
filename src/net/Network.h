/* This header file is writen by qp09
 * usually just for fun
 * Sat October 24 2015
 */
#ifndef NETWORK_H
#define NETWORK_H

#include <assert.h>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

#include "../utils/SimInfo.h"
#include "../interface/Neuron.h"
#include "../interface/Synapse.h"
#include "../interface/ModelView.h"
#include "DistriNetwork.h"

using std::pair;
using std::find;
using std::array;
using std::vector;
using std::map;
using std::set;

typedef ModelView<Neuron> Population;
typedef ModelView<Synapse> Projection;

class Network {
public:
	Network(int node_num = 1);
	~Network();

	int set_node_num(int node_num); 
	
	// template<class N>
	// Population<N>* createNeuron(N n1);
	template<class N>
	Population* createPopulation(int id, size_t num, N templ);

	template<class N>
	Population* createPopulation(size_t num, N templ);

	template<class S>
	int connect(Population *pSrc, Population *pDst, S templ, SpikeType sp = Exc);
	template<class S>
	int connect(Population *p_src, size_t src, Population *p_dst, size_t dst, S syn, SpikeType s_type);

	int connect(Population *p_src, Population *p_dst, real weight, real delay, SpikeType type);
	int connect(Population *p_src, size_t src, Population *p_dst, size_t dst, real weight, real delay, real tau=0);
	int connect(Population *p_src, Population *p_dst, real *weight, real *delay, SpikeType *type, size_t size);

	// int connectOne2One(Population *pSrc, Population *pDst, real *weight, real *delay, SpikeType *type, size_t size);
	// int connectConv(Population *pSrc, Population *pDst, real *weight, real *delay, SpikeType *type, size_t height, size_t width, size_t k_height, size_t k_width);
	// int connectPooling(Population *pSrc, Population *pDst, real weight, size_t height, size_t width, size_t p_height, size_t p_width);
	
	int connect(ID src, ID dst, ID syn, unsigned int delay);
	
	// int connect(size_t populationIDSrc, size_t neuronIDSrc, size_t populationIDDst, size_t neuronIDDst, real weight, real delay, real tau = 0);
	// Synapse* connect(Neuron *pSrc, Neuron *pDst, real weight, real delay, SpikeType type = Excitatory, real tau = 0, bool store = true);

	GNetwork* buildNetwork(const SimInfo &info);

	int reset(const SimInfo &info);

	void logMap();
	void status();

	DistriNetwork * buildNetworks(const SimInfo &info, bool auto_splited = true);

	CrossThreadData * arrangeCrossThreadData(int node_num);
	CrossThreadDataGPU * arrangeCrossThreadDataGPU(int node_num);

	CrossNodeData * arrangeCrossNodeData(int node_num, const SimInfo &info);



private:
	void mapIDtoIdx(GNetwork *net);
	// bool checkIDtoIdx();
	
	void splitNetwork();
	void countTypeNum();
	GNetwork* arrangeData(int node, const SimInfo &info);
	Connection* arrangeConnect(size_t n_num, size_t s_num, int node_idx, const SimInfo &info);
	CrossNodeMap* arrangeCrossNodeMap(size_t n_num, int node_idx, int node_num);

public:
	/** Cross Node Data **/
	// map<ID, int> _nID2node;
	// map<ID, int> _sID2node;
	// Neurons that on this node and would issue spikes to others.
	// Acessed by neurons = _crossnodeNeuronsSend[node]
	vector<set<Neuron *> > _crossnodeNeuronsSend;
	// Neurons that on other nodes and would issue spike to this node.
	// Accessed by neurons = _crossnodeNeuronsRecv[node]
	vector<set<Neuron *> > _crossnodeNeuronsRecv;
	// Get idx of shadow neuron on destination node, the idxs of shadow neurons are larger than that of real neurons.
	// Accessed by idx = _crossnodeNeuron2idx[node][neuron]
	vector<map<Neuron *, int> > _crossnodeNeuron2idx;

	/** Per Node Data **/
	// vector<map<int, ID> > _global_idx2nID;
	// vector<map<int, ID> > _global_idx2sID;
	// Number of neurons for different types on different nodes accessed by _global_ntype_num[node][type]
	vector<map<Type, unsigned long long> >	_globalNTypeNum;
	// Number of synapses for different types on different nodes accessed by _global_ntype_num[node][type]
	vector<map<Type, unsigned long long> > _globalSTypeNum;

	map<Type, Neuron*> _neurons;
	map<Type, Synapse*> _synapses;
	vector<Population *> _populations;

	map<ID, map<unsigned int, vector<ID>>> n2s_conn;
	map<ID, ID> s2n_conn;

	unsigned int _max_delay;
	unsigned int _min_delay;

	uint64_t _neuron_num;
	uint64_t _synapse_num;
	int _node_num;
private:
	real _maxFireRate;
	vector<Type> _nTypes;
	vector<Type> _sTypes;
	vector<int> _neuronNums;
	vector<int> _connectNums;
	vector<int> _synapseNums;
};

template<class N>
Population * Network::createPopulation(int id, size_t num, N templ)
{
	return createPopulation(num, templ);
}

template<class N>
Population * Network::createPopulation(size_t num, N templ)
{
	//ID id = totalNeuronNum;
	Population *pp1 = NULL;

	Type type = templ.get_type();

	if (_neurons.find(type) == _neurons.end()) {
		N *neuron = new N(templ, num);
		_neurons[type] = neuron;
		pp1 = new Population(neuron, 0, num);
	} else {
		size_t num_t = _neurons[type]->size();
		N *neuron = _neurons[type];
		neuron->append(&templ, num);
		pp1 = new Population(neuron, num_t, num);
	}

	_populations.push_back(pp1);

	return pp1;
}

template<class S>
int Network::connect(Population *p_src, Population *p_dst, S templ, SpikeType sp) {
	size_t src_size = p_src->size();
	size_t dst_size = p_dst->size();

	size_t size = src_num * dst_num;

	Type type = templ.type();
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new S(templ, size);
	} else {
		offset = _synapses.size();
		_synapses[type]->append(&templ, size);
	}

	int count = 0;
	for (size_t s=0; s<src_num; s++) {
		for (size_t d =0; d<dst_num; d++) {
			size_t s_offset = offset + s * dst_num + d;
			connect(ID(p_src->type(), 0, p_src->offset()+s), ID(p_dst->type(), sp, p_dst->offset()+d), ID(type, 0, s_offset), _synapses[type]->get_delay()[s_offset]);
			count++;
		}
	}

	return count;
}

template<class S>
int Network::connect(Population *p_src, size_t src, Population *p_dst, size_t dst, S templ, SpikeType sp)
{

	Type type = templ.type();
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new S(templ);
	} else {
		offset = _synapses.size();
		_synapses[type]->append(&templ);
	}

	connect(ID(p_src->type(), 0, p_src->offset()+src), ID(p_dst->type(), sp, p_dst->offset()+dst), ID(type, 0, offset), _synapses[type]->get_delay()[offset]);

	return 1;
}

#endif /* NETWORK_H */

