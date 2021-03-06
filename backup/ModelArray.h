/* This header file is writen by qp09
 * usually just for fun
 * Fri October 23 2015
 */
#ifndef MODEL_ARRAY_H
#define MODEL_ARRAY_H

#include <vector>

#include "../third_party/json/json.h"
#include "../interface/Model.h"
#include "../interface/Synapse.h"

using std::vector;

template<class Item>
class ModelArray : public Model {
public:
	ModelArray(size_t n, Type type, int node = 0);

	template<class N>
	ModelArray(size_t n, const N &templ);

	template<class N>
	ModelArray(int id, size_t n, const N &templ);

	~ModelArray();

	virtual Type getType() const override;

	size_t getNum();
	// virtual size_t getSize();

	virtual int getNode() final;
	virtual void setNode(int node) final;
	// virtual int reset(SimInfo &info);
	// virtual int update(SimInfo &info);
	// virtual void monitor(SimInfo &info);

	// virtual int getData(void *data);
	virtual int hardCopy(void *data, int idx, int base, const SimInfo &info) override;

	int append(Item *templ);
	// Item* findItem(ID id);
	// Item* getItem(int idx);
	Item* find(ID id);
	Item* locate(int idx);
	
// protected:
	size_t _N;
	Type _type;
	vector<Item*> _items;
};

// typedef ModelArray<Neuron> Population;
// typedef ModelArray<Synapse> Projection;

template<class Item>
ModelArray<Item>::ModelArray(size_t n, Type type, int node) : Model(node)
{
	_items.reserve(n);
	_type = type;
	_N = n;
}

template<class Item>
template<class N>
ModelArray<Item>::ModelArray(size_t n, const N &templ) : Model()
{
	this->_node = templ.getNode();
	_type = templ.getType();
	_N = n;
	_items.resize(n, NULL);
	for(unsigned int i=0; i<_N; i++) {
		Item * p = new N(templ);
		_items[i] = p;
	}
}

template<class Item>
template<class N>
ModelArray<Item>::ModelArray(ID id, size_t n, const N &templ) : Model(templ.node)
{
	printf("The initialization of id is depracated, the ids are currently auto managed\n");
	ModelArray(n, templ);
}

template<class Item>
ModelArray<Item>::~ModelArray() 
{
	for(auto iter=_items.begin(); iter!=_items.end(); iter++) {
		Item *p = *iter;
		delete p;
	}
	_items.clear();
	_N = 0;
}

template<class Item>
size_t ModelArray<Item>::getNum()
{
	return _N;
}

template<class Item>
int ModelArray<Item>::append(Item *templ)
{
	assert(_type == templ->getType());
	_items.push_back(templ);
	if (_items.size() > _N) {
		printf("ModelArray overflow!!");
		_N = _items.size();
	}

	return _items.size() - 1;
}

template<class Item>
Item* ModelArray<Item>::locate(int idx)
{
	return _items[idx];
}

template<class Item>
Item* ModelArray<Item>::find(ID id)
{
	for (unsigned int i=0; i< _items.size(); i++) {
		if (_items[i]->getID() == id) {
			return _items[i];
		}
	}

	return NULL;
}

template<class Item>
int ModelArray<Item>::getNode()
{
	printf("getnode for population departured, dont use it!\n");
	return _node;
}

template<class Item>
void ModelArray<Item>::setNode(int node)
{
	Model::setNode(node);
	typename vector<Item>::iterator iter;
	for (auto iter = _items.begin(); iter != _items.end(); iter++) {
		(*iter)->setNode(node);
	}
	printf("setnode for population departured, dont use it!\n");
}

// template<class Item>
// void ModelArray<Item>::monitor(SimInfo &info)
// {
// 	if(monitored) {
// 		typename vector<Item>::iterator iter;
// 		for (iter = items.begin(); iter != items.end(); iter++) {
// 			iter->monitorOn();
// 			iter->monitor(info);
// 		}
// 	}
// }

// template<class Item>
// int ModelArray<Item>::update(SimInfo &info)
// {
// 	int ret = 0;
// 	typename vector<Item>::iterator iter;
// 	for (iter = items.begin(); iter != items.end(); iter++) {
// 		ret += iter->update(info);
// 	}
// 	return ret;
// }
// 
// template<class Item>
// int ModelArray<Item>::reset(SimInfo &info)
// {
// 	int ret = 0;
// 	typename vector<Item>::iterator iter;
// 	for (iter = items.begin(); iter != items.end(); iter++) {
// 		ret += iter->reset(info);
// 	}
// 	return ret;
// }

// template<class Item>
// size_t ModelArray<Item>::getSize()
// {
// 	if (items.size() > 0) {
// 		return N*items[0].getSize();
// 	} else {
// 		return 0;
// 	}
// }

template<class Item>
int ModelArray<Item>::hardCopy(void *data, int idx, int base, const SimInfo &info)
{
	size_t copiedIdxs = 0;
	for (auto iter = _items.begin(); iter != _items.end(); iter++) {
		size_t copied = (*iter)->hardCopy(data, idx+copiedIdxs, base, info);
		copiedIdxs += copied;
	}
	return copiedIdxs;
}

template<class Item>
Type ModelArray<Item>::getType() const 
{
	return _type;
}

// template<class Item>
// int ModelArray<Item>::getData(void *data) {
// 	Json::Value *p = (Json::Value *)data;
// 	(*p)["id"] = getID();
// 	(*p)["size"] = (unsigned int)N;
// 	(*p)["neuron_type"] = 0;
// 	Json::Value parameters(Json::arrayValue);
// 	typename vector<Item>::iterator iter;
// 	for (iter = items.begin(); iter != items.end(); iter++) {
// 		Json::Value parameter;
// 		iter->getData(&parameter);
// 		parameters.append(parameter);
// 	}
// 	(*p)["parameters"] = parameters;
// 
// 	return 0;
// }

#endif /* ModelArray_H */

