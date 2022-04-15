
#include <map>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#include "include/metis.h"
#include "helper_c.h"
#include "mem.h"

using std::map;
using std::pair;
using std::find;
using std::vector;
using std::string;
using std::unordered_map;
using std::unordered_set;

int to_metis(const char *graph_name, const char *res1_name, const char *res2_name, const char *res_name, int nparts1, int nparts2)
{

	size_t neuron_num = 0;
	FILE *graph = fopen_c(graph_name, "rb");
	fread_c(&neuron_num, 1, graph);
	fclose_c(graph);

	FILE *res1_f = fopen_c(res1_name, "rb");
	idx_t r1_size = 0;
	fread_c(&r1_size, 1, res1_f);
	idx_t *res1 = new idx_t[r1_size];
	fread_c(res1, r1_size, res1_f);

	idx_t *res = new idx_t[neuron_num];
	std::fill(res, res+neuron_num, 1000000);

	string g_name(graph_name);
	string r2_name(res2_name);

	for (int i=0; i<nparts1; i++) {
		size_t size2 = 0;

		FILE *fr = fopen_c((g_name+"_rev"+std::to_string(i)).c_str(), "rb");
		fread_c(&size2, 1, fr);
		size_t *rev_map = new size_t[size2];
		fread_c(rev_map, size2, fr);
		fclose_c(fr);

		FILE *f2 = fopen_c((r2_name+"_"+std::to_string(i)).c_str(), "rb");
		idx_t r2_size = 0;
		fread_c(&r2_size, 1, f2);
		idx_t *res2 = new idx_t[r2_size];
		fread_c(res2, r2_size, f2);
		fclose_c(f2);

		for (size_t j=0; j<r2_size; j++) {
			assert(res[rev_map[j]] == 1000000);
			res[rev_map[j]] = res1[rev_map[j]] * nparts2 + res2[j];
		}

		delete [] rev_map;
		delete [] res2;
	}

	for (size_t i=0; i<neuron_num; i++) {
		assert(res[i] < nparts1 * nparts2);
	}

	FILE *res_f = fopen_c(res_name, "wb+");
	fwrite_c(&neuron_num, 1, res_f);
	fwrite_c(res, neuron_num, res_f);
	fclose_c(res_f);

	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 7) {
		printf("Usuage: %s graph_origin res1_file res2_file output_file nparts1 nparts2\n", argv[0]);
		exit(-1);
	}
	to_metis(argv[1], argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));
}
