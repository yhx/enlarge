
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

int to_metis(const char *in_name, const char *out_name)
{
	size_t neuron_num = 0;
	size_t synapse_num = 0;

	FILE *in = fopen_c(in_name, "rb");

	fread_c(&neuron_num, 1, in);
	fread_c(&synapse_num, 1, in);

	vector<size_t> tmp;
	vector<vector<size_t>> vec_conn;
	vec_conn.resize(neuron_num);

	printf("Start %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);
	for (size_t n=0; n<neuron_num; n++) {
		size_t size = 0;
		fread_c(&size, 1, in);
		if (size > 0) {
			tmp.resize(size);
			fread_c(tmp.data(), size, in);
			for (size_t v=0; v<size; v++) {
				vec_conn[n].push_back(tmp[v]);
				vec_conn[tmp[v]].push_back(n);
			}
		}
	}
	printf("Finish load data %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);

	idx_t *xadj = new idx_t[neuron_num+1];
	idx_t *adjncy = new idx_t[synapse_num*2];

	size_t s_offset = 0;
	xadj[0] = 0;
	for (size_t n=0; n<neuron_num; n++) {
		for (auto i=vec_conn[n].begin(); i!=vec_conn[n].end(); i++) {
			size_t tid = *i;
			adjncy[s_offset] = tid;
			s_offset++;
		}
		xadj[n+1] = s_offset;
	}
	printf("%ld %ld\n", s_offset, synapse_num);
//	assert(s_offset == 2*synapse_num);

	printf("Finish prepare data\n");

	FILE *out = fopen_c(out_name, "wb+");
	fwrite_c(&neuron_num, 1, out);
	fwrite_c(&synapse_num, 1, out);
	fwrite_c(xadj, neuron_num+1, out);
	fwrite_c(adjncy, synapse_num*2, out);
	printf("Finish write data\n");

	fclose_c(out);

	delete[] xadj;
	delete[] adjncy;

	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		printf("Usuage: to_metis inputfile outputfile\n");
		exit(-1);
	}

	return to_metis(argv[1], argv[2]);
}
