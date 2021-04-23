
#include <map>
#include <vector>
#include <algorithm>

#include "../src/utils/helper_c.h"

using std::vector;
using std::map;
using std::find;

int to_metis(const char *in_name, const char *out_name)
{
	vector<map<unsigned int, vector<size_t>>> conn;
	FILE *in = fopen_c(in_name, "r");
	FILE *out = fopen_c(out_name, "w+");
	size_t neuron_num;
	size_t synapse_num;

	fread_c(&neuron_num, 1, in);
	fread_c(&synapse_num, 1, in);

	conn.resize(neuron_num);
	for (size_t n=0; n<neuron_num; n++) {
		size_t nid = 0;
		size_t count_t = 0;
		size_t d_s = 0;
		unsigned int delay_t = 0;
		fread_c(&nid, 1, in);
		assert(nid == n);
		fread_c(&d_s, 1, in);
		for (size_t i=0; i<d_s; i++) {
			fread_c(&delay_t, 1, in);
			fread_c(&count_t, 1, in);

			if (count_t > 0) {
				conn[n][delay_t].resize(count_t);
				fread_c(conn[n][delay_t].data(), count_t, in);
			}
		}
	}

	size_t syn_num = synapse_num;
	for (size_t n=0; n<neuron_num; n++) {
		for (auto di=conn[n].begin(); di!=conn[n].end(); di++) {
			for (auto ti=di->second.begin(); ti!=di->second.end(); ti++) {
				size_t tid = *ti;
				if (find(conn[tid][di->first].begin(), conn[tid][di->first].end(), n) == conn[tid][di->first].end()) {
					conn[tid][di->first].push_back(n);
				} else {
					syn_num--;
				}
			}
		}
	}

	fprintf(out, "%ld %ld\n", neuron_num, syn_num);
	for (size_t n=0; n<neuron_num; n++) {
		for (auto di=conn[n].begin(); di!=conn[n].end(); di++) {
			for (auto ti=di->second.begin(); ti!=di->second.end(); ti++) {
				size_t tid = *ti;
				fprintf(out, "%ld %u ", tid, di->first);
			}
		}
		fprintf(out, "\n");
	}

	// fseek(out, 0, SEEK_SET);
	// fprintf(out, "%ld %ld\n", neuron_num, synapse_num);

	fclose_c(in);
	fclose_c(out);

	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Usuage: to_metis inputfile\n");
	}


	return to_metis(argv[1], "grpah.metis");
}
