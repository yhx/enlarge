
#include <cstdint>

#include "type.h"

// struct DetailID {
// 	uint32_t id;
// 	uint16_t pre;
// 	uint8_t offset;
// 	uint8_t type;
// };
// 
// union UnionID
// {
// 	DetailID detail;
// 	uint64_t id;
// };

const int TYPE_BITS = 8;
const int OFFSET_BITS = 8;

const int64_t TYPE_CAST     = 0xFF;
const int64_t OFFSET_CAST   = 0xFF;

const int64_t TYPE_UNMASK     = 0xFF00000000000000;
const int64_t OFFSET_UNMASK   = 0x00FF000000000000;

const int64_t TYPE_MASK   = 0x00FFFFFFFFFFFFFF;
const int64_t OFFSET_MASK = 0xFF00FFFFFFFFFFFF;

class ID {
public:
	ID() {
		_id = 0;
	}
	ID(uint64_t id) {
		_id = id;
	}
	ID(Type type,  uint64_t offset, uint64_t id) {
		_id = id;
		set_type(type);
		set_offset(offset);
	}
	~ID() {}

	void set_type(Type type) {
		uint64_t t = (type & TYPE_CAST);
		_id = (_id & TYPE_MASK) + ((t<<(64-TYPE_BITS)) & TYPE_UNMASK);
	}

	int get_type() {
		return ((_id >> (64-TYPE_BITS)) & TYPE_CAST);
	}

	void set_offset(int offset) {
		uint64_t t = offset & 0xFF;
		_id = (_id & OFFSET_MASK) + ((t<<(64-TYPE_BITS-OFFSET_BITS)) & OFFSET_UNMASK);
	}

	int get_offset() {
		return ((_id >> (64-TYPE_BITS-OFFSET_BITS)) & OFFSET_CAST);
	}

	// UnionID _id;
	uint64_t _id;
};
