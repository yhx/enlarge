
#include <map>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#include "include/metis.h"
#include "helper_c.h"
#include "mem.h"

using std::vector;
using std::pair;
using std::map;
using std::unordered_map;
using std::unordered_set;
using std::find;

int to_metis(const char *org_name, const char *in_name, const char *out_name, int grp_size)
{
	size_t org_n_num = 0;

	size_t neuron_num = 0;
	size_t synapse_num = 0;

	FILE *org = fopen_c(org_name, "rb");
	FILE *in = fopen_c(in_name, "rb");
	FILE *out = fopen_c(out_name, "wb+");


	fread_c(&org_n_num, 1, org);

	fread_c(&neuron_num, 1, in);

	idx_t *org_res = new idx_t[org_n_num];
	idx_t *in_res = new idx_t[neuron_num];
	
	fread_c(in_res, neuron_num, in);

	for (size_t i=0; i<org_n_num; i++) {
		org_res[i] = in_res[i/grp_size];
	}


	fwrite_c(&org_n_num, 1, out);
	fwrite_c(org_res, org_n_num, out);
	printf("Finish write data\n");

	fclose_c(org);
	fclose_c(in);
	fclose_c(out);

	delete [] org_res;
	delete [] in_res;

	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 5) {
		printf("Usuage: %s org_file grp_res_file output_file grp_size\n", argv[0]);
		exit(-1);
	} else {
		return to_metis(argv[1], argv[2], argv[3], atoi(argv[4]));
	}

}
