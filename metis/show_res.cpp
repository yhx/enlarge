
#include "include/metis.h"
#include "helper_c.h"

int show_res(const char *in_name)
{
	FILE *in = fopen_c(in_name, "r");

	idx_t nvtxs = 0;
	fread_c(&nvtxs, 1, in);
	idx_t *part = new idx_t[nvtxs];
	fread_c(part, nvtxs, in);

	printf("%ld\n", nvtxs);
	for (size_t n=0; n<(size_t)nvtxs; n++) {
		printf("%ld ", part[n]);
	}
	printf("\n");

	fclose_c(in);

	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Usuage: %s inputfile\n", argv[0]);
		exit(-1);
	}

	return show_res(argv[1]);
}
