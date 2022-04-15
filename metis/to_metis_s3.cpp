
#include <map>
#include <vector>
#include <unordered_set>
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
using std::unordered_set;
using std::find;

int to_metis(const char *in_name, const char *out_name, idx_t nparts, const char *w_name = NULL)
{
	idx_t *vwgt = NULL;
	size_t n_t = 0;

	if (w_name) {	

		FILE * wf = fopen_c(w_name, "r");
		fread_c(&n_t, 1, wf);
		size_t *id_t = new size_t[n_t];
		size_t *w_t = new size_t[n_t];
		fread_c(id_t, n_t, wf);
		fread_c(w_t, n_t, wf);

		fclose_c(wf);

		vwgt = new idx_t[n_t];

		for (size_t i=0; i<n_t; i++) {
			vwgt[id_t[i]] = w_t[i];
		}

		delete [] id_t;
		delete [] w_t;
	} 

	size_t neuron_num = 0;
	size_t synapse_num = 0;

	FILE *in = fopen_c(in_name, "rb");
	FILE *out = fopen_c(out_name, "wb+");

	fread_c(&neuron_num, 1, in);
	fread_c(&synapse_num, 1, in);

	assert(!w_name || neuron_num == n_t);

	idx_t *xadj = new idx_t[neuron_num+1];
	idx_t *adjncy = new idx_t[synapse_num*2];
	idx_t *adjwgt = NULL;

	fread_c(xadj, neuron_num+1, in);
	fread_c(adjncy, synapse_num*2, in);

	printf("Finish prepare data\n");

	idx_t nvtxs = neuron_num;
	idx_t ncon = 1;
	idx_t *part = new idx_t[nvtxs];
	idx_t objval = 0;

	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	// options[METIS_OPTION_NSEPS] = 10;
	// options[METIS_OPTION_UFACTOR] = 100;
	options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
	options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
	options[METIS_OPTION_NUMBERING] = 0;



	struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL); 

	METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, NULL, NULL, options, &objval, part);
	
	gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;

    printf("Stage 3 time : %lf \n",timeuse);	
	printf("Finish split graph\n");


	// fprintf(out, "%ld %ld\n", neuron_num, synapse_num);
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
	if (argc != 4 && argc != 5) {
		printf("Usuage: to_metis input_file output_file parts [weight_file]\n");
		exit(-1);
	} else if (argc == 5) {
		return to_metis(argv[1], argv[2], atoi(argv[3]), argv[4]);
	} else {
		return to_metis(argv[1], argv[2], atoi(argv[3]));
	}

}
