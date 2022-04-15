
#include <map>
#include <vector>
#include <set>
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
using std::set;
using std::fill;

typedef vector<map<unsigned int, vector<size_t>>> CONN_D;
typedef vector<unordered_set<size_t>> CONN_UD;

int Load_graph(const char *graphFile,CONN_UD &ud_conn,size_t &neu_num, size_t &syn_num,int group_size,vector<idx_t> &vwgt,vector<idx_t> &awgt)
{
    // printf("Start %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);
	printf("start load graph...\n");
	size_t neuron_num;
	size_t synapse_num;

	FILE *in = fopen_c(graphFile, "rb");

	CONN_D conn;

	fread_c(&neuron_num, 1, in);
	fread_c(&synapse_num, 1, in);
	printf("neuron_num : %d   synapse num : %d\n",neuron_num,synapse_num);

	// vector<idx_t> n2s;
	// n2s.resize(neuron_num);
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
	printf("Load finished!\n");
	// printf("Finish load data %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);



    size_t n_num = 0, s_num = 0;
	n_num = (neuron_num+group_size-1)/group_size;

	ud_conn.resize(n_num);
	
	// vwgt = new idx_t[n_num];
	// awgt = new idx_t[n_num*n_num];
	// memset(vwgt,0,sizeof(idx_t)*n_num);
	// memset(awgt,0,sizeof(idx_t)*n_num*n_num);

	vwgt.resize(n_num);
	awgt.resize(n_num*n_num);

	fill(vwgt.begin(),vwgt.end(),0);
	fill(awgt.begin(),awgt.end(),0);

	size_t rs= 0;
	for (size_t n=0; n<neuron_num; n++) {
		for (auto di=conn[n].begin(); di!=conn[n].end(); di++) {
			for (auto ti=di->second.begin(); ti!=di->second.end(); ti++) {
				size_t tid = *ti;
				// ud_conn[n].insert((tid, di->first));
                tid = tid / group_size;
                size_t nid = n / group_size;
				vwgt[tid] += 1;

				// TODO:
				awgt[nid*n_num + tid] += 1;
				awgt[tid*n_num + nid] += 1;


				if(nid == tid)
					rs ++;
				else if (nid < tid) {
                    if(ud_conn[nid].find(tid) == ud_conn[nid].end())
                    {
                        ud_conn[nid].insert(tid);
                        ud_conn[tid].insert(nid);
						// awgt[nid*n_num + tid] += 1;
                        s_num ++;
                    }
					
				} else {
                    if(ud_conn[tid].find(nid) == ud_conn[tid].end())
                    {
                        ud_conn[nid].insert(tid);
                        ud_conn[tid].insert(nid);
						// awgt[nid*n_num + tid] += 1;
                        s_num ++;
                    }
				}
			}
		}
	}
    syn_num = s_num;
	neu_num = neuron_num;
	// printf("Finish simplify data %lf\n", getCurrentRSS()/1024.0/1024.0/1024.0);
	printf("Construct graph finished!  %ld\n",rs);
	

	// debug 
	// size_t sum = 0;
	// for(size_t i =0 ;i < n_num; i++)
	// 	sum += ud_conn[i].size();
	// printf("===================== : %ld + %ld\n",sum,rs);
    return 0;

}

int preprocess(CONN_UD &ud_conn,size_t syn_num, vector<idx_t> &xadj,vector<idx_t> &adjncy, vector<idx_t> &awgt,vector<idx_t> &adjwgt)
{
    size_t neu_num = ud_conn.size();
    
    // xadj = new idx_t[neu_num + 1];
    // adjncy = new idx_t[syn_num * 2];
	// adjwgt = new idx_t[syn_num * 2];
	xadj.resize(neu_num+1);
	adjncy.resize(syn_num*2);
	adjwgt.resize(syn_num*2);

    xadj[0] = 0;
    size_t s_offset = 0;

    for (size_t n=0; n<neu_num; n++) {
		for (auto i=ud_conn[n].begin(); i!=ud_conn[n].end(); i++) {
			size_t tid = *i;
			adjncy[s_offset] = tid;
			adjwgt[s_offset] = awgt[n*neu_num+tid];
			s_offset++;
		}
		xadj[n+1] = s_offset;
	}
	printf("s_offset : %d   syn_num : %d \n",s_offset,syn_num);
    // assert(s_offset == 2*syn_num);

    return 0;
}

int split(idx_t nvtxs, idx_t nparts, vector<idx_t> xadj, vector<idx_t> adjncy, idx_t *part,  vector<idx_t> vwgt, vector<idx_t> adjwgt,idx_t vw = 1)
{

	printf("Start split...\n");
	idx_t objval = 0;

	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	// options[METIS_OPTION_NSEPS] = 10;
	// options[METIS_OPTION_UFACTOR] = 100;
	options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
	options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
	options[METIS_OPTION_NUMBERING] = 0;

//	METIS_PartGraphKway(&nvtxs, &vw, xadj.data(), adjncy.data(), vwgt.data(), NULL, adjwgt.data(), &nparts, NULL, NULL, options, &objval, part);
	METIS_PartGraphKway(&nvtxs, &vw, xadj.data(), adjncy.data(), vwgt.data(), NULL, NULL, &nparts, NULL, NULL, options, &objval, part);
	printf("Finish split graph\n");
    return 0;
}

int Part(int npart1,int npart2,size_t neu_num,size_t syn_num,idx_t *parts,CONN_UD &ud_conn,vector<idx_t> &awgt,vector<idx_t> &vwgt,idx_t *res_part,idx_t vw)
{
	printf("start part....\n");
	vector<set<idx_t> > sub_parts;
	sub_parts.resize(npart1);
	for(size_t i=0; i < neu_num; i++)
	{
		sub_parts[parts[i]].insert(i);
	}

	// part graph

	CONN_UD sub_graph;
	map<idx_t,idx_t> converter;
	map<idx_t,idx_t> inv_conv;

	
	idx_t *tmp_part = new idx_t[neu_num];
	vector<idx_t> sub_adjncy, sub_adjwgt, sub_xadj,sub_vwgt;
	
	sub_adjncy.resize(syn_num*2);
	sub_adjwgt.resize(syn_num*2);
	sub_xadj.resize(neu_num+1);
	sub_vwgt.resize(neu_num);


	for(int i=0; i<npart1; i++)
	{
		idx_t part_size = sub_parts[i].size();
		printf("Parting %d  of size  %ld ...\n",i,part_size);
		sub_graph.clear();
		sub_graph.resize(part_size);

		memset(tmp_part,0,sizeof(idx_t)*neu_num);

		
		

		converter.clear();
		inv_conv.clear();
		idx_t idx = 0;
		for(auto p:sub_parts[i])
		{
			converter[p] = idx;
			inv_conv[idx] = p;
			idx ++;
		}

		assert(idx == part_size);
		printf(" Start construct...\n");
		sub_xadj[0] = 0;
		idx_t s_offset = 0;
		for(auto p:sub_parts[i])
		{
			for (auto it=ud_conn[p].begin(); it!=ud_conn[p].end(); it++) {
				idx_t tid = *it;
				// if tid in sub_parts[i]
				if(sub_parts[i].find(tid) != sub_parts[i].end())
				{
					sub_adjncy[s_offset] = converter[tid];
					sub_adjwgt[s_offset] = awgt[p*neu_num+tid];
					s_offset++;
				}
				
			}
			sub_xadj[converter[p]+1] = s_offset;
			sub_vwgt[converter[p]] = vwgt[p];
		}
		// printf(" s_offset : %ld  xadj[%ld] : %ld\n",s_offset,part_size,sub_xadj[part_size]);

		split(part_size,npart2,sub_xadj,sub_adjncy,tmp_part,sub_vwgt,sub_adjwgt,vw);

		// repart
		for(idx_t j=0;j<part_size;j++)
		{
			res_part[inv_conv[j]] = i * npart2 + tmp_part[j];
		}

		printf("Part %d finished!\n",i);
	}

	delete [] tmp_part;

	return 0;
}

int main(int argc, char **argv)
{
    // if (argc != 3) {
	// 	printf("Usuage: to_metis inputfile outputfile\n");
	// 	exit(-1);
	// }

    // graph :  argv[1];
	// outfile : argv[2]
	// npart1 : argv[3]
	// group_size : argv[4]
	// npart2 : argv[5]


	idx_t vw = 1;
    

	if(argc < 4)
	{
		printf("Parameters aren't enough!\n");
		exit(-1);
	}
	int npart1 = atoi(argv[3]);
	
	int group_size = 1;
    int npart2 = 1;
	if(argc == 5)
	{
		group_size = atoi(argv[4]);
	}
	else
	{
		group_size = atoi(argv[4]);
		npart2 = atoi(argv[5]);
	}
    

    CONN_UD ud_conn;
    size_t syn_num = 0;
    size_t neu_num = 0;
	idx_t  n_num = 0;

	vector<idx_t> vwgt, awgt, adjwgt;
	vector<idx_t> xadj, adjncy;
	
	// get group graph
    Load_graph(argv[1],ud_conn,neu_num,syn_num,group_size,vwgt,awgt);
	n_num = ud_conn.size();	

	printf("neuron num : %d   n num : %d  synapse num : %d\n",neu_num,n_num,syn_num);

	// process to metis format
	preprocess(ud_conn,syn_num,xadj,adjncy,awgt,adjwgt);

	idx_t *part = new idx_t[n_num];
	idx_t *res_part = new idx_t[n_num];

	memset(part,0,sizeof(idx_t)*n_num);
	memcpy(res_part,part,sizeof(idx_t)*n_num);

	split(n_num,npart1,xadj,adjncy,part,vwgt,adjwgt,vw);

	// 2 level part
	if(npart2 > 1)
	{
		memset(res_part,0,sizeof(idx_t)*n_num);
		Part(npart1,npart2,n_num,syn_num,part,ud_conn,awgt,vwgt,res_part,vw);
	}

	// divide group
	idx_t *res = new idx_t[neu_num];
	for(size_t i =0; i < neu_num; i++)
	{
		res[i] = res_part[i/group_size];
	}
	// for(size_t i=0; i < n_num; i++)
	// 	printf("%ld\n",res_part[i]);

	FILE *out = fopen_c(argv[2], "wb+");
	fwrite_c(&neu_num, 1, out);
	fwrite_c(res, neu_num, out);
	printf("Finish write data\n");
	fclose_c(out);

	delete [] res;
	delete [] part;
	delete [] res_part;

	return 0;

}
