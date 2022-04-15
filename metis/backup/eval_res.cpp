
#include <map>

#include "include/metis.h"
#include "helper_c.h"

using std::map;

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Usuage: %s inputfile\n", argv[0]);
		exit(-1);
	}

	FILE *in = fopen_c(argv[1], "r");

	map<idx_t, size_t> m;

	idx_t nvtxs = 0;
	fread_c(&nvtxs, 1, in);
	idx_t *part = new idx_t[nvtxs];
	fread_c(part, nvtxs, in);

	for (size_t i=0; i<nvtxs; i++) {
		m[part[i]]++;
	}

	for (auto iter = m.begin(); iter != m.end(); iter++) {
		printf("%ld: %lu\n", iter->first, iter->second);
	}

	fclose_c(in);

	return 0;
}
