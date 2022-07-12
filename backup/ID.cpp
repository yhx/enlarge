
#include <cstdio>

#include "ID.h"

using namespace std;

int main() {
	ID id;
	id.id = 10000;

	printf("%lu\n", id.id);
	printf("%d %d %d %d\n", id.detail.type, id.detail.offset, id.detail.pre, id.detail.id);
	printf("0x%016lx\n", id.id);
	printf("0x%02x %02x %04x %08x\n", id.detail.type, id.detail.offset, id.detail.pre, id.detail.id);

	id.detail.type = 1;

	printf("%lu\n", id.id);
	printf("%d %d %d %d\n", id.detail.type, id.detail.offset, id.detail.pre, id.detail.id);
	printf("0x%016lx\n", id.id);
	printf("0x%02x %02x %04x %08x\n", id.detail.type, id.detail.offset, id.detail.pre, id.detail.id);

	
	return 0;
};



