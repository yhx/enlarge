
#include <cstdint>

struct Detail {
	uint32_t id;
	uint16_t pre;
	uint8_t offset;
	uint8_t type;
};

union ID
{
	Detail detail;
	uint64_t id;
};
