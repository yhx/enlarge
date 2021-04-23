
#include <map>
#include <vector>

#include "../src/utils/helper_c.h"

using std::vector;
using std::map;

int to_metis(const char *in_name, const char *out_name)
{
	vector<map<unsigned int, vector<size_t>>> conn;
	FILE *in = fopen_c(in_name);
	FILE *out = fopen_c(out_name);
	size_t _neuron_num;
	size_t _synapse_num;

	fread_c(&_neuron_num, 1, in);
	fread_c(&_synapse_num, 1, in);

	conn.resize(_neuron_num);
	for (size_t n=0; n<_neuron_num; n++) {
		size_t nid = 0;
		size_t count_t = 0;
		unsigned int delay_t = 0;
		fread_c(&nid, 1, in);
		assert(nid == n);
		fread_c(&delay_t, 1, in);
		fread_c(&count_t, 1, in);
		
		conn[n][delay_t].resize(count_t);
		fread_c(conn[n][delay_t].data(), count_t, in);
	}



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
