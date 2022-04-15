
#include <map>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sys/time.h>

#include "include/metis.h"
#include "helper_c.h"
#include "mem.h"

using std::vector;
using std::pair;
using std::map;
using std::unordered_map;
using std::find;

int to_metis(const char *in_name, const char *out_name, idx_t nparts)
{
	size_t neuron_num;
	size_t synapse_num;
	size_t syn_num = 0;
	FILE *out = fopen_c(out_name, "w+");

	vector<map<unsigned int, vector<size_t>>> conn;
	vector<unordered_map<size_t, unsigned int>> ud_conn;
	vector<vector<pair<size_t, unsigned int>>> vec_conn;

	printf("Start %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);
	FILE *in = fopen_c(in_name, "r");

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
	fclose_c(in);
	printf("Finish load data %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);

	syn_num = synapse_num;
	ud_conn.resize(neuron_num);

	// size_t nid =0;
	for (size_t n=0; n<neuron_num; n++) {
		for (auto di=conn[n].begin(); di!=conn[n].end(); di++) {
			for (auto ti=di->second.begin(); ti!=di->second.end(); ti++) {
				size_t tid = *ti;
				// ud_conn[n].insert((tid, di->first));
				if (n <= tid) {
					if (ud_conn[n].find(tid) != ud_conn[n].end()) {
						ud_conn[n][tid] = (ud_conn[n][tid] + di->first)/2;
						syn_num--;
					} else {
						ud_conn[n][tid] = di->first;
					}
				} else {
					if (ud_conn[tid].find(n) != ud_conn[tid].end()) {
						ud_conn[tid][n] = (ud_conn[tid][n] + di->first)/2;
						syn_num--;
					} else {
						ud_conn[tid][n] = di->first;
					}
				}
			}
		}
	}
	printf("Finish simplify data %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);

	for (size_t n=0; n<neuron_num; n++) {
		for (auto di=conn[n].begin(); di!=conn[n].end(); di++) {
			di->second.clear();
			vector<size_t>().swap(di->second);
		}
	}
	// conn.clear();
	vector<map<unsigned int, vector<size_t>>>().swap(conn);
	printf("Clear vec %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);

	vec_conn.resize(neuron_num);
	for (size_t n=0; n<neuron_num; n++) {
		for (auto i=ud_conn[n].begin(); i!=ud_conn[n].end(); i++) {
			vec_conn[n].push_back(std::make_pair(i->first, i->second));
			vec_conn[i->first].push_back(std::make_pair(n, i->second));
		}
	}
	printf("Finish convert data %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);
	ud_conn.clear();
	vector<unordered_map<size_t, unsigned int>>().swap(ud_conn);
	printf("Clear vec %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);


	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	// options[METIS_OPTION_NSEPS] = 10;
	// options[METIS_OPTION_UFACTOR] = 100;
	options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
	options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
	options[METIS_OPTION_NUMBERING] = 0;

	idx_t *xadj = new idx_t[neuron_num+1];
	idx_t *adjncy = new idx_t[syn_num*2];
	idx_t *adjwgt = new idx_t[syn_num*2];

	// size_t n_offset = 1;
	size_t s_offset = 0;
	xadj[0] = 0;
	for (size_t n=0; n<neuron_num; n++) {
		for (auto i=vec_conn[n].begin(); i!=vec_conn[n].end(); i++) {
			size_t tid = i->first;
			adjncy[s_offset] = tid;
			adjwgt[s_offset] = i->second;
			s_offset++;
		}
		xadj[n+1] = s_offset;
	}
	printf("%ld %ld\n", s_offset, syn_num);
	assert(s_offset == 2*syn_num);

	idx_t nvtxs = neuron_num;
	idx_t ncon = 1;
	idx_t *part = new idx_t[nvtxs];
	idx_t objval = 0;
	printf("Finish prepare data\n");

	METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, adjwgt, &nparts, NULL, NULL, options, &objval, part);
	printf("Finish split graph\n");

	// fprintf(out, "%ld %ld\n", neuron_num, syn_num);
	// for (size_t n=0; n<neuron_num; n++) {
	// 	for (auto di=conn[n].begin(); di!=conn[n].end(); di++) {
	// 		for (auto ti=di->second.begin(); ti!=di->second.end(); ti++) {
	// 			size_t tid = *ti;
	// 			fprintf(out, "%ld %u ", tid+1, di->first);
	// 		}
	// 	}
	// 	fprintf(out, "\n");
	// }

	// fseek(out, 0, SEEK_SET);
	// fprintf(out, "%ld %ld\n", neuron_num, synapse_num);
	fwrite_c(&nvtxs, 1, out);
	fwrite_c(part, nvtxs, out);
	printf("Finish write data\n");

	fclose_c(out);

	delete[] xadj;
	delete[] adjncy;
	delete[] adjwgt;
	delete[] part;

	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 4) {
		printf("Usuage: to_metis inputfile outputfile parts\n");
		exit(-1);
	}

	return to_metis(argv[1], argv[2], atoi(argv[3]));
}
