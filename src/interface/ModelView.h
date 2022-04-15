/* This header file is writen by qp09
 * usually just for fun
 * Fri October 23 2015
 */
#ifndef MODEL_VIEW_H
#define MODEL_VIEW_H

#include "../interface/Model.h"

template<class Item>
class ModelView {
public:
	ModelView(Item *item, size_t offset, size_t num);
	~ModelView();

	Type type() const;
	size_t offset() const;
	size_t size() const;

private:
	size_t _offset;
	size_t _size;
	Item* _items;
};

template<class Item>
ModelView<Item>::ModelView(Item *item, size_t offset, size_t num)
{
	_items = item;
	_offset = offset;
	_size = num;
}

template<class Item>
ModelView<Item>::~ModelView() 
{
	_items = NULL;
	_offset = 0;
	_size = 0;
}

template<class Item>
size_t ModelView<Item>::size() const
{
	return _size;
}

template<class Item>
size_t ModelView<Item>::offset() const 
{
	return _offset;
}


template<class Item>
Type ModelView<Item>::type() const 
{
	return _items->type();
}

#endif /* ModelView_H */

