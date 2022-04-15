/* This header file is writen by qp09
 * usually just for fun
 * Mon February 22 2016
 */
#ifndef MODEL_H
#define MODEL_H

#include <map>

#include "../utils/SimInfo.h"
#include "../base/ID.h"

using std::map;

class Model {
public:
	Model() : _num(0), _type(UNSET), _id() {}

	Model(Type type, size_t num, size_t offset=0, int buffer_size=0) : _num(num), _type(type), _buffer_size(buffer_size) {
		_id.set_type(type);
		_id.set_offset(offset);
	}

	virtual ~Model() { _num=0; }

	inline size_t size () const {
		return _num;
	}

	inline Type type() const {
		return _type;
	}

	inline ID get_ID() const {
		return _id;
	}

	inline void set_ID(ID id) {
		this->_id = id;
	}

	inline int buffer_size() {
		return _buffer_size;
	}


	virtual void * packup() = 0;
	virtual int packup(void *data, size_t dst, size_t src) = 0;
	
protected:
	size_t _num;
	Type _type;
	ID _id;
	int _buffer_size;
};

#endif /* MODEL_H */

