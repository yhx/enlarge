/* This header file is writen by qp09
 * usually just for fun
 * Mon February 22 2016
 */
#ifndef MODEL_H
#define MODEL_H

#include <map>

#include "../utils/SimInfo.h"
#include "../utils/ID.h"

using std::map;

class Model {
public:
	Model() : _num(0), _type(UNSET), _id() {}

	Model(Type type, int num, int offset=0) : _num(num), _type(type) {
		_id.set_type(type);
		_id.set_offset(offset);
	}

	virtual ~Model() { _num=0; }

	inline int size () const {
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


	virtual int packup(void *data) = 0;
	
protected:
	size_t _num;
	Type _type;
	ID _id;
};

#endif /* MODEL_H */

