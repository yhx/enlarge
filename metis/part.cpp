
#include <map>
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include <algorithm>

#include "include/metis.h"
#include "helper_c.h"
#include "mem.h"

using std::vector;
using std::string;
using std::pair;
using std::map;
using std::unordered_map;
using std::set;
using std::find;

int to_metis(const char *graph_name, const char *part_name, const int nparts)
{
	size_t neuron_num = 0;
	size_t synapse_num = 0;

	FILE *graph = fopen_c(graph_name, "rb");

	fread_c(&neuron_num, 1, graph);
	fread_c(&synapse_num, 1, graph);

	vector<size_t> tmp;
	vector<set<size_t>> graph_org;
	graph_org.resize(neuron_num);

	printf("Start %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);
	for (size_t n=0; n<neuron_num; n++) {
		size_t size = 0;
		fread_c(&size, 1, graph);
		if (size > 0) {
			tmp.resize(size);
			fread_c(tmp.data(), size, graph);
			for (size_t v=0; v<size; v++) {
				graph_org[n].insert(tmp[v]);
			}
		}
	}
	tmp.clear();
	vector<size_t>(tmp).swap(tmp);
	fclose_c(graph);
	printf("Finish load data %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);

	FILE *part_f = fopen_c(part_name, "rb");

	vector<set<size_t>> part_org;
	part_org.resize(nparts);

	idx_t p_size = 0;
	fread_c(&p_size, 1, part_f);
	idx_t *p_tmp = new idx_t[p_size];
	fread_c(p_tmp, p_size, part_f);

	for (size_t i=0; i<p_size; i++) {
		assert(p_tmp[i] < nparts);
		part_org[p_tmp[i]].insert(i);
	}

	fclose_c(part_f);
	delete [] p_tmp;

	size_t n_sum = 0;
	vector<vector<size_t>> subgraph;
	string name(graph_name);
	for (int p=0; p<nparts; p++) {
		size_t t_s = part_org[p].size();
		size_t *rev_map = new size_t[t_s];
		FILE *f = fopen_c((name+"_sub"+std::to_string(p)).c_str(), "wb+");

		size_t count = 0;
		size_t count_s = 0;

		fwrite_c(&t_s, 1, f);
		subgraph.resize(t_s);

		unordered_map<size_t, size_t> converter;

		for (auto const &elem : part_org[p]) {
			subgraph[count].resize(std::min(t_s, graph_org[elem].size()));
			auto iter = set_intersection(part_org[p].begin(), part_org[p].end(), graph_org[elem].begin(), graph_org[elem].end(), subgraph[count].begin());
			size_t offset = std::distance(subgraph[count].begin(), iter);
			subgraph[count].resize(offset);
			rev_map[count] = elem;
			converter[elem] = count;
			count++;
			count_s += offset;
		}

		fwrite_c(&count_s, 1, f);
		FILE *f2 = fopen_c((name+"_rev"+std::to_string(p)).c_str(), "wb+");
		fwrite_c(&t_s, 1, f2);
		fwrite_c(rev_map, t_s, f2);
		fclose_c(f2);

		

		for (auto &elem :subgraph) {
			for (size_t i=0; i<elem.size(); i++) {
				elem[i] = converter[elem[i]];
			}
		}

		for (size_t i=0; i<t_s; i++) {
			size_t tmp = subgraph[i].size();
			fwrite_c(&tmp, 1, f);
			fwrite_c(subgraph[i].data(), tmp, f);
		}
		fclose(f);
		delete [] rev_map;

		n_sum += count;
	}

	assert(n_sum == neuron_num);

	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 4) {
		printf("Usuage: %s origin_graph part_res part_num\n", argv[0]);
		exit(-1);
	}

	return to_metis(argv[1], argv[2], atoi(argv[3]));
}
