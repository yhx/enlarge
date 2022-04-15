
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


int main(int argc, char **argv)
{
	if (argc != 4) {
		printf("Usuage: to_metis inputfile outputfile group_size\n");
		exit(-1);
	}

	printf("Start %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);
	size_t neuron_num;
	size_t synapse_num;

	int group_size = atoi(argv[3]);

	FILE *in = fopen_c(argv[1], "rb");

	vector<unordered_set<size_t>> ud_conn;
	vector<map<unsigned int, vector<size_t>>> conn;


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


	size_t n_num = 0, s_num = 0;
	n_num = (neuron_num+group_size-1)/group_size;
	ud_conn.resize(n_num);

	for (size_t n=0; n<neuron_num; n++) {
		for (auto di=conn[n].begin(); di!=conn[n].end(); di++) {
			for (auto ti=di->second.begin(); ti!=di->second.end(); ti++) {
				size_t tid = *ti;
				tid = tid / group_size;
				size_t nid = n / group_size;
				if(nid == tid)
					printf("fuck\n");
				if (nid <= tid) {
					if (ud_conn[nid].find(tid) == ud_conn[nid].end()) {
						ud_conn[nid].insert(tid);
						s_num++;
					}
				} else {
					if (ud_conn[tid].find(nid) == ud_conn[tid].end()) {
						ud_conn[tid].insert(nid);
						s_num++;
					}
				}
			}
		}
	}

	size_t sum = 0;
	for(size_t i =0 ;i < n_num; i++)
		sum += ud_conn[i].size();

	printf("===================== : %ld\n",sum);

	printf("Finish simplify data %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);
	printf("=====> %ld , %ld \n",n_num,s_num);
	FILE *out = fopen_c(argv[2], "wb+");
	fwrite_c(&n_num, 1, out);
	fwrite_c(&s_num, 1, out);
	size_t counter = 0;
	for (size_t n=0; n<n_num; n++) {
	// for (auto  n=ud_conn.begin(); n!=ud_conn.end(); n++) {
		size_t size = ud_conn[n].size();
		fwrite_c(&size, 1, out);
		for (auto ti=ud_conn[n].begin(); ti!=ud_conn[n].end(); ti++) {
			counter++;
			size_t tid = *ti;
			fwrite_c(&tid, 1, out);
		}
	}
	fclose(out);

	assert(counter == s_num);
}
