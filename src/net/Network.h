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

#include "../utils/Para.h"
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
typedef map<int, map<Type, size_t>> CrossTypeInfo_t;
typedef map<int, size_t> CrossInfo_t;

class Network {
/**
 * used to store the connections and neurons info.
 **/
public:
	Network(real dt, int node_num = 1);
	~Network();

	int set_node_num(int node_num); 
	
	// template<class N>
	// Population<N>* createNeuron(N n1);
	template<class N>
	Population* createPopulation(int id, size_t num, N templ);

	template<class N>
	Population* createPopulation(size_t num, N templ);

	// add `Type type` for all connect() to support poisson connection
	template<class S>
	int connect(Population *pSrc, Population *pDst, S templ, SpikeType sp = Exc);
	template<class S>
	int connect(Population *p_src, size_t src, Population *p_dst, size_t dst, S syn, SpikeType sp);

	int connect_(ID src, ID dst, ID syn, unsigned int delay);

	int connect(Population *p_src, Population *p_dst, real weight, real delay, real tau, SpikeType sp=Exc);
	int connect(Population *p_src, Population *p_dst, real *weight, real *delay, real *tau, SpikeType *sp, size_t size);
	int connect(Population *p_src, size_t src, Population *p_dst, size_t dst, real weight, real delay, real tau, SpikeType=Exc);

	int connect(Population *p_src, Population *p_dst, real *weight, real *delay, real tau, SpikeType sp, size_t size);
	int connect(Population *p_src, Population *p_dst, real *weight, real *delay, SpikeType *sp, size_t size);

	int connect(Population *p_src, Population *p_dst, int *idx_src, int *idx_dst, int synapse_num, real *weight, real *delay, SpikeType *sp);
	/**
	 * connect_poisson_generator用于给目的神经元p_dst连接上poisson generator。
	 */
	int connect_poisson_generator_(ID dst, ID syn, unsigned int delay);
	int connect_poisson_generator(Population *p_dst, real *mean, real *weight, real *delay, SpikeType *sp);

	// int connectOne2One(Population *pSrc, Population *pDst, real *weight, real *delay, SpikeType *type, size_t size);
	// int connectConv(Population *pSrc, Population *pDst, real *weight, real *delay, SpikeType *type, size_t height, size_t width, size_t k_height, size_t k_width);
	// int connectPooling(Population *pSrc, Population *pDst, real weight, size_t height, size_t width, size_t p_height, size_t p_width);
	
	int reset(const SimInfo &info);

	void logMap();
	void log_graph();
	void save_graph(const char *name="n2n.graph");
	void status();

	GNetwork* buildNetwork(const SimInfo &info);
	DistriNetwork * buildNetworks(const SimInfo &info, SplitType split = SynapseBalance, const char *name="", const AlgoPara *para = NULL, bool auto_splited = true);

	// CrossThreadData * arrangeCrossThreadData(int node_num);
	CrossThreadDataGPU * arrangeCrossGPUData();

	CrossNodeData * arrangeCrossNodeData(const SimInfo &info);

private:
	// void mapIDtoIdx(GNetwork *net);
	// bool checkIDtoIdx();
	
	size_t add_type_conn(Type type, size_t size);
	
	void splitNetwork(SplitType split, const char *name="", const AlgoPara *para = NULL);
	void update_status();
	void update_status_splited();


	int arrangeNet(DistriNetwork *net, CrossTypeInfo_t &type_offset, CrossTypeInfo_t &_neuron_offset, CrossTypeInfo_t &neuron_count, CrossTypeInfo_t &synapse_offset, CrossTypeInfo_t &synapse_count);
	
	int arrangeNeuron(DistriNetwork *net, CrossTypeInfo_t &type_offset, CrossTypeInfo_t &neuron_offset, CrossTypeInfo_t &neuron_count);

	int arrangeLocal(DistriNetwork *net, CrossTypeInfo_t & type_offset, CrossTypeInfo_t &neuron_offset, CrossTypeInfo_t & synapse_offset, CrossTypeInfo_t &neuron_count, CrossTypeInfo_t &synapse_count, CrossTypeInfo_t &n2s_count, int delay);

	int arrangeCross(DistriNetwork *net, CrossTypeInfo_t & type_offset, CrossTypeInfo_t &synapse_count, CrossTypeInfo_t &n2s_count, CrossInfo_t &cross_idx, CrossInfo_t &node_n_offset, int delay);

	// GNetwork* arrangeData(int node, const SimInfo &info);
	// Connection* arrangeConnect(size_t n_num, size_t s_num, int node_idx, const SimInfo &info);
	// CrossNodeMap* arrangeCrossNodeMap(size_t n_num, int node_idx, int node_num);

public:
	/**
	 * `_poisson_synapse2delay` is used only in poisson synapses. 
	 * `_poisson_synapse2delay[i]` is the delay of poisson synapse's ID.
	 * It used to generate dst attribute in class `Connection`.
	 **/
	// vector<map<unsigned int, vector<ID>>> _poisson_synapse2dst;
	// vector<delay, vector<unsigned int>> _poisson_synapse2dst;
	/** Cross Node Data **/
	map<Type, vector<int>> _idx2node;
	// map<ID, int> _nid2node;
	// map<ID, int> _sid2node;

	CrossTypeInfo_t _neuron_nums;
	CrossTypeInfo_t _synapse_nums;
	CrossTypeInfo_t _buffer_offsets;
	// map<unsigned int, map<Type, size_t>> _neuron_nums;
	// map<unsigned int, map<Type, size_t>> _synapse_nums;
	map<ID, size_t> _id2node_idx;
	// Neurons that on this node and would issue spikes to others.
	// Acessed by neurons = _crossnodeNeuronsSend[node]
	vector<set<ID> > _crossnodeNeuronsSend;
	// Neurons that on other nodes and would issue spike to this node.
	// Accessed by neurons = _crossnodeNeuronsRecv[node]
	vector<set<ID> > _crossnodeNeuronsRecv;
	// Get idx of shadow neuron on destination node, the idxs of shadow neurons are larger than that of real neurons.
	// Accessed by idx = _crossnodeNeuron2idx[node][neuron]
	// vector<map<Neuron *, int> > _crossnodeNeuron2idx;

	/** Per Node Data **/
	// vector<map<int, ID> > _global_idx2nID;
	// vector<map<int, ID> > _global_idx2sID;
	// Number of neurons for different types on different nodes accessed by _global_ntype_num[node][type]
	// vector<map<Type, unsigned long long> >	_globalNTypeNum;
	// Number of synapses for different types on different nodes accessed by _global_ntype_num[node][type]
	// vector<map<Type, unsigned long long> > _globalSTypeNum;

	// use polymorphic to store specific neurons and synapses info
	map<Type, Neuron*> _neurons;  // 根据神经元类型type获取对应类型所有神经元
	map<Type, Synapse*> _synapses;  // 根据突触类型type获取对应类型所有突触
	vector<Population *> _populations;

	/**
	 * _conn_n2s[src.type()][src.id()][delay].push_back(syn);
	 * 根据Type获得'神经元到突触的连接'
	 * 通过源神经元和delay获得对应的突触
	 * 前提：已经获得了源神经元的类型、源神经元的id、突触延迟->得到这个源神经元的所有突触
	 **/
	map<Type, vector<map<unsigned int, vector<ID>>>> _conn_n2s; 
	/**
	 * conn_s2n[syn.type()][syn.id()] = dst;
	 * 根据Type获取对应的突触连接ID的向量
	 * 通过突触类型得到突触连接的目的节点
	 * 前提：已经获得了突触的类型和突触的id的信息->得到突触连接的目的神经元
	 */
	map<Type, vector<ID>> _conn_s2n;

	/**
	 * _conn_sd2n[syn.type()][delay][syn.id()] = dst;
	 * 根据突触类型、突触id以及delay获取目的神经元ID
	 **/
	map<Type, map<int, vector<pair<int, ID> > > > _conn_sd2n;
	// map<ID, map<unsigned int, vector<ID>>> n2s_conn;
	// map<ID, ID> s2n_conn;

	// map<ID, map<unsigned int, vector<ID>>> n2s_conn_rev;
	// map<ID, ID> s2n_conn_rev;

	int _max_delay;
	int _min_delay;

	// uint64_t _neuron_num;
	// uint64_t _synapse_num;
	size_t _neuron_num;
	size_t _synapse_num;
	map<Type, size_t> _neurons_offset;
	map<Type, size_t> _synapses_offset;
	int _node_num;
	real _dt;
// private:
// 	real _maxFireRate;
// 	vector<Type> _nTypes;
// 	vector<Type> _sTypes;
// 	vector<int> _neuronNums;
// 	vector<int> _connectNums;
// 	vector<int> _synapseNums;
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

	Type type = templ.type();

	if (_neurons.find(type) == _neurons.end()) {
		N *neuron = new N(templ, num);
		_neurons[type] = neuron;
		pp1 = new Population(neuron, 0, num);
	} else {
		size_t num_t = _neurons[type]->size();
		N *neuron = dynamic_cast<N*>(_neurons[type]);
		neuron->append(&templ, num);
		pp1 = new Population(neuron, num_t, num);
	}
	add_type_conn(type, num);

	_populations.push_back(pp1);
	return pp1;
}

template<class S>
int Network::connect(Population *p_src, Population *p_dst, S templ, SpikeType sp) {
	size_t src_size = p_src->size();
	size_t dst_size = p_dst->size();

	size_t size = src_size * dst_size;

	Type type = templ.type();
	size_t offset = 0;
	if (_synapses.find(type) == _synapses.end()) {
		_synapses[type] = new S(templ, size);
	} else {
		offset = _synapses[type]->size();
		_synapses[type]->append(&templ, size);
	}
	add_type_conn(type, size);

	int count = 0;
	for (size_t s=0; s<src_size; s++) {
		for (size_t d =0; d<dst_size; d++) {
			size_t s_offset = offset + s * dst_size + d;
			connect_(ID(p_src->type(), 0, p_src->offset()+s), ID(p_dst->type(), sp, p_dst->offset()+d), ID(type, 0, s_offset), _synapses[type]->delay()[s_offset]);
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
		offset = _synapses[type]->size();
		_synapses[type]->append(&templ);
	}
	add_type_conn(type, 1);

	connect_(ID(p_src->type(), 0, p_src->offset()+src), ID(p_dst->type(), sp, p_dst->offset()+dst), ID(type, 0, offset), _synapses[type]->delay()[offset]);

	return 1;
}

#endif /* NETWORK_H */

