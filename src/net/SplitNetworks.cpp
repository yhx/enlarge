
#include <float.h>
#include <sys/time.h>
#include "../utils/utils.h"
#include "Network.h"

void swap_id(ID &s, ID &d, map<Type, vector<vector<ID>>> &n2s_rev, map<Type, vector<int>> &idx2node, int src, int dst) 
{
	Type t = s.type();
	size_t id = s.id();
	idx2node[t][id] = dst;
	for (auto siter = n2s_rev[t][id].begin(); siter != n2s_rev[t][id].end(); siter++) {
		idx2node[siter->type()][siter->id()] = dst;
	}

	t = d.type();
	id = d.id();
	idx2node[d.type()][d.id()] = src;
	for (auto siter = n2s_rev[t][id].begin(); siter != n2s_rev[t][id].end(); siter++) {
		idx2node[siter->type()][siter->id()] = src;
	}
}

void Network::splitNetwork(SplitType split, const char *name, const AlgoPara *para)
{
	if (_node_num <= 1) {
		for (auto iter = _neurons.begin(); iter != _neurons.end(); iter++) {
			_idx2node[iter->first].resize(iter->second->size(), 0);
		}

		for (auto iter = _synapses.begin(); iter != _synapses.end(); iter++) {
			_idx2node[iter->first].resize(iter->second->size(), 0);
		}
		return;
	}

	print_mem("before n2s_rev");
	map<Type, vector<vector<ID>>> n2s_rev;
	for (auto iter = _neurons.begin(); iter != _neurons.end(); iter++) {
		n2s_rev[iter->first].resize(iter->second->size());
		_idx2node[iter->first].resize(iter->second->size(), -1);
	}

	for (auto iter = _synapses.begin(); iter != _synapses.end(); iter++) {
		_idx2node[iter->first].resize(iter->second->size(), -1);
	}

	for (auto ti = _conn_s2n.begin(); ti != _conn_s2n.end(); ti++) {
		for (size_t idx = 0; idx < ti->second.size(); idx++) {
			ID &t = ti->second[idx];
			n2s_rev[t.type()][t.id()].push_back(ID(ti->first, idx));
		}
	}
	print_mem("after n2s_rev");


	struct timeval t1,t2,t3,t4;
    double timeuse,timeno = 0;
    gettimeofday(&t1,NULL);
	switch (split) {
		case NeuronBalance:
			{
				// printf("===NEU_BASE\n");
				int node_idx = 0;
				size_t neuron_count = 0;
				size_t neuron_per_node = _neuron_num/_node_num;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);
						_idx2node[t][i] = node_idx;
						neuron_count += 1;
						gettimeofday(&t3,NULL);					
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}
						gettimeofday(&t4,NULL);
						timeno += (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
						if (neuron_count>= (node_idx+1) * neuron_per_node && node_idx < _node_num - 1) {
							node_idx++;	
						}
					}
				}
			}
			break;
		case RoundRobin:
			{
				// printf("===ROUND_ROBIN\n");
				size_t neuron_count = 0;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);
						int node_idx = neuron_count % _node_num;
						_idx2node[t][i] = node_idx;
						neuron_count += 1;

						gettimeofday(&t3,NULL);	
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}
						gettimeofday(&t4,NULL);
						timeno += (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
					}
				}
			}
			break;
		case GrpRR:
			{
				// printf("Group RoundRobin\n");
				size_t neuron_count = 0;
				size_t node_idx = 0;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);

						_idx2node[t][i] = node_idx;
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}

						if (neuron_count % 32 == 31) {
							node_idx = (node_idx+1) % _node_num;
							neuron_count = 0;
						} else {
							neuron_count += 1;
						}
					}
				}
			}
			break;
		case SynBestFit:
			{
				// printf("Synapse Bestfit\n");
				vector<int> neu_count(_node_num, 0);
				vector<int> syn_count(_node_num, 0);
				const int alpha = 100;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);

						int node_idx = 0;
						int count = INT_MAX;

						for (int node = 0; node < _node_num; node++) {
							int v = neu_count[node] * alpha + syn_count[node];
							if (count > v) {
								count = v;
								node_idx = node;
							}
						}

						_idx2node[t][i] = node_idx;
						neu_count[node_idx]++;
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}
						syn_count[node_idx] += n2s_rev[t][i].size();
					}
				}
			}
			break;
		case Metis:
			{
				string s(name);
				string s1 = s + ".metis";
				FILE *f = fopen(s1.c_str(), "rb");
				if (!f) {
					string w1 = s + ".weight";
					FILE *wf = fopen_c(w1.c_str(), "w+");
					fwrite_c(&_neuron_num, 1, wf);
					size_t count = 0;
					for (auto ti = n2s_rev.begin(); ti != n2s_rev.end(); ti++) {
						for (size_t idx = 0; idx < ti->second.size(); idx++) {
							size_t nid = _neurons_offset[ti->first]+idx;
							fwrite_c(&nid, 1, wf);
							count++;
						}
					}
					assert(count == _neuron_num);
					count = 0;
					for (auto ti = n2s_rev.begin(); ti != n2s_rev.end(); ti++) {
						for (size_t idx = 0; idx < ti->second.size(); idx++) {
							size_t s_t = n2s_rev[ti->first][idx].size();
							fwrite_c(&s_t, 1, wf);
							count++;
						}
					}
					assert(count == _neuron_num);
					fclose_c(wf);

					string s2 = s + ".graph";
					printf("===Metis 0\n");
					save_graph(s2.c_str());
					exit(0);
				} else {
					// printf("===Metis\n");
					size_t n_num = 0;
					fread_c(&n_num, 1, f);
					assert(n_num == _neuron_num);
					size_t node_idx = 0;
					for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
						Type t = t_iter->first;
						for (size_t i=0; i<t_iter->second->size(); i++) {
							fread_c(&node_idx, 1, f);
							_idx2node[t][i] = node_idx;
							for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
								_idx2node[siter->type()][siter->id()] = node_idx;
							}
						}
					}
					fclose_c(f);
				}
			}
			break;
		case BestFit:
			{
				float syn_weight = 0.05;
				float comm_weight = 0.2;
				float send_weight = 1.5;
				float recv_weight = 0.5;
				
				if (para) {
					syn_weight = para->syn_weight;
					comm_weight = para->comm_weight;
					send_weight = para->comm_weight;
					recv_weight = para->recv_weight;
				}

				// printf("===BestFit %.4lf %.4lf %.4lf %.4lf\n", syn_weight, comm_weight, send_weight, recv_weight);
				// print_mem("before s2n_rev");
				map<Type, vector<ID>> s2n_rev;
				for (auto iter = _synapses.begin(); iter != _synapses.end(); iter++) {
					s2n_rev[iter->first].resize(iter->second->size(), -1);
				}

				for (auto ti = _conn_n2s.begin(); ti != _conn_n2s.end(); ti++) {
					for (size_t idx = 0; idx < ti->second.size(); idx++) {
						for (auto di = ti->second[idx].begin(); di != ti->second[idx].end(); di++) {
							for (auto si = di->second.begin(); si != di->second.end(); si++) {
								s2n_rev[si->type()][si->id()] = ID(ti->first, idx);
							}
						}
					}
				}
				print_mem("after s2n_rev");

				float *load = malloc_c<float>(_node_num);
				float *comm = malloc_c<float>(_node_num);
				float *comm_idx = malloc_c<float>(_node_num);

				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						float burden = n2s_rev[t][i].size() * syn_weight + 1;

						float value = FLT_MAX;
						unsigned int node_idx = 0;

						for (int idx = 0; idx < _node_num; idx++) {
							float c_cost = 0.0;
							memset_c(comm, 0, _node_num);
							for (auto i_i = _conn_n2s[t][i].begin(); i_i != _conn_n2s[t][i].end(); i_i++) {
								for (auto s_i = i_i->second.begin(); s_i != i_i->second.end(); s_i++) {
									ID &nid = _conn_s2n[s_i->type()][s_i->id()];
									int nidx = _idx2node[nid.type()][nid.id()];
									if (nidx >= 0) { 
										if (idx != nidx) {
											comm[nidx] += recv_weight;
											comm[idx] += send_weight;
											c_cost += recv_weight + send_weight;
										} else {
											comm[nidx] += comm_weight;
											c_cost += comm_weight;
										}
									}
								}
							}

							for (auto si = n2s_rev[t][i].begin(); si != n2s_rev[t][i].end(); si++) {
								ID &nid = s2n_rev[si->type()][si->id()];
								int nidx = _idx2node[nid.type()][nid.id()];
								if (nidx >= 0) {
								   	if (idx != nidx) {
										comm[nidx] += send_weight;
										comm[idx] += recv_weight;
										c_cost += recv_weight + send_weight;
									} else {
										comm[nidx] += comm_weight;
										c_cost += comm_weight;
									}
								}
							}

							float eval = load[idx] + burden + c_cost;
							if (eval < value) {
								value = eval;
								node_idx = idx;
								memcpy_c(comm_idx, comm, _node_num);
							}
						}
						_idx2node[t][i] = node_idx;
						load[node_idx] += burden;
						for (int i=0; i<_node_num; i++) {
							load[i] += comm_idx[i];
						}
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}
					}
				}
				free_c(load);
				free_c(comm);
				free_c(comm_idx);
				print_mem("before clear s2n_rev");
				s2n_rev.clear();
				print_mem("after clear s2n_rev");
			}
			break;
		case Balanced:
			{
				printf("===BALANCED\n");
				const int MAX_LOOP = 1000;
				float syn_weight = 0.1;
				float comm_weight = 30;
				float send_weight = 1;
				float *load = malloc_c<float>(_node_num);
				float *comm = malloc_c<float>(_node_num);
				float *comm_idx = malloc_c<float>(_node_num);

				print_mem("before s2n_rev");
				map<Type, vector<ID>> s2n_rev;
				for (auto iter = _synapses.begin(); iter != _synapses.end(); iter++) {
					s2n_rev[iter->first].resize(iter->second->size(), -1);
				}

				for (auto ti = _conn_n2s.begin(); ti != _conn_n2s.end(); ti++) {
					for (size_t idx = 0; idx < ti->second.size(); idx++) {
						for (auto di = ti->second[idx].begin(); di != ti->second[idx].end(); di++) {
							for (auto si = di->second.begin(); si != di->second.end(); si++) {
								s2n_rev[si->type()][si->id()] = ID(ti->first, idx);
							}
						}
					}
				}
				print_mem("after s2n_rev");

				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						float burden = n2s_rev[t][i].size() * syn_weight + 1;

						float value = FLT_MAX;
						unsigned int node_idx = 0;

						for (int idx = 0; idx < _node_num; idx++) {
							float c_cost = 0.0;
							memset_c(comm, 0, _node_num);
							for (auto i_i = _conn_n2s[t][i].begin(); i_i != _conn_n2s[t][i].end(); i_i++) {
								for (auto s_i = i_i->second.begin(); s_i != i_i->second.end(); s_i++) {
									ID &nid = _conn_s2n[s_i->type()][s_i->id()];
									int nidx = _idx2node[nid.type()][nid.id()];
									if (nidx >= 0 && idx != nidx) {
										comm[nidx] += 1;
										comm[idx] += send_weight;
										c_cost += 1 + send_weight;
									}
								}
							}

							for (auto si = n2s_rev[t][i].begin(); si != n2s_rev[t][i].end(); si++) {
								ID &nid = s2n_rev[si->type()][si->id()];
								int nidx = _idx2node[nid.type()][nid.id()];
								if (nidx >= 0 && idx != nidx) {
									comm[nidx] += send_weight;
									comm[nidx] += 1;
									c_cost += 1 + send_weight;
								}
							}

							float eval = load[idx] + burden + c_cost *comm_weight;
							if (eval < value) {
								value = eval;
								node_idx = idx;
								memcpy_c(comm_idx, comm, _node_num);
							}
						}
						_idx2node[t][i] = node_idx;
						load[node_idx] += burden;
						for (int i=0; i<_node_num; i++) {
							load[i] += comm_idx[i];
						}
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}
					}
				}

				map<Type, int *> comm_log;
				for (auto iter = _neurons.begin(); iter != _neurons.end(); iter++) {
					comm_log[iter->first] = malloc_c<int>(iter->second->size() * _node_num);
				}

				vector<ID> v_id(_node_num);
				vector<int> v_value(_node_num, INT_MIN);
				vector<int> v_nidx(_node_num, -1);
				for (int loop = 0; loop < MAX_LOOP; loop++) {
					for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
						Type t = t_iter->first;
						for (size_t i=0; i<t_iter->second->size(); i++) {
							int nidx = _idx2node[t][i];
							assert(nidx > 0);
							for (auto d_i = _conn_n2s[t][i].begin(); d_i != _conn_n2s[t][i].end(); d_i++) {
								for (auto s_i = d_i->second.begin(); s_i != d_i->second.end(); s_i++) {
									int sidx = _idx2node[s_i->type()][s_i->id()];
									comm_log[t][i*_node_num + sidx]++;
								}
							}

							for (auto s_i = n2s_rev[t][i].begin(); s_i != n2s_rev[t][i].end(); s_i++) {
								int sidx = _idx2node[s_i->type()][s_i->id()];
								comm_log[t][i*_node_num + sidx]++;
							}

							for (int n = 0; n < _node_num; n++) {
								if (n != nidx) {
									int d_ = comm_log[t][i*_node_num + n] - comm_log[t][i*_node_num + nidx];
									if (d_ > v_value[nidx]) {
										v_value[nidx] = d_;
										v_nidx[nidx] = n;
										v_id[nidx] = ID(t, i);
									}
								}
							}
						}
					}
					bool update = false;
					for (int n = 0; n < _node_num; n++) {
						int n_d = v_nidx[n];
						if (n_d >= 0 && v_value[n] > 0) {
							update = true;
							if (v_nidx[n_d] == n) {
								swap_id(v_id[n], v_id[n_d], n2s_rev, _idx2node, n, n_d);
							} else {
								int d_value = INT_MIN;
								ID d_id(0);
								for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
									Type t = t_iter->first;
									for (size_t i=0; i<t_iter->second->size(); i++) {
										int d_ = comm_log[t][i*_node_num + n] - comm_log[t][i*_node_num - n_d];
										if (d_ > d_value) {
											d_value = d_;
											d_id = ID(t, i);
										}
									}
								}
								if (d_value > -v_value[n]) {
									swap_id(v_id[n], d_id, n2s_rev, _idx2node, n, n_d);
								}
							}
						}
					}

					if (!update) {
						printf("LOOP: %d\n", loop);
						break;
					}
				}


				free_c(load);
				free_c(comm);
				free_c(comm_idx);
				print_mem("before clear s2n_rev");
				s2n_rev.clear();
				print_mem("after clear s2n_rev");
			}
			break;
		// case NSynB:
		// 	{
		// 		printf("===New synapse balance:\n");
		// 		size_t exc_num = 0;
		// 		size_t inh_num = 0;
		// 		for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		// 			Type t = t_iter->first;
		// 			for (size_t i=0; i<t_iter->second->size(); i++) {
		// 				// t i
		// 				for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
		// 					// printf("weight: ");
		// 					// printf("%lf \n",_synapses[siter->type()]->weight(siter->id()));
		// 					if(_synapses[siter->type()]->weight(siter->id()) >= 0)
		// 					{
		// 						exc_num ++;
		// 					}
		// 					else
		// 					{
		// 						inh_num ++;
		// 					}
		// 				}
		// 			}
		// 		}
		// 		printf("exc / inh : %ld  %ld \n",exc_num,inh_num);
		// 		size_t exc_per_node = (exc_num -1)/_node_num + 1;
		// 		size_t inh_per_node = (inh_num - 1) / _node_num + 1;
		// 		size_t exc_cnt = 0;
		// 		size_t inh_cnt = 0;
		// 		// size_t exc_idx, inh_idx;
		// 		// exc_idx = inh_idx =0;
		// 		// for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		// 		// 	Type t = t_iter->first;
		// 		// 	for (size_t i=0; i<t_iter->second->size(); i++) {
		// 		// 		ID id(t,0,i);
						
		// 		// 		for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
		// 		// 			if(_synapses[siter->type()]->weight(siter->id()) >= 0)
		// 		// 			{
		// 		// 				if(exc_cnt + 1 > exc_per_node && exc_idx < _node_num-1)
		// 		// 				{
		// 		// 					exc_idx ++;
		// 		// 					exc_cnt = 0;
		// 		// 				}
		// 		// 				exc_cnt ++;
		// 		// 				_idx2node[t][i] = exc_idx;
		// 		// 				_idx2node[siter->type()][siter->id()] = exc_idx;
		// 		// 			}
		// 		// 			else
		// 		// 			{
		// 		// 				if(inh_cnt + 1 > inh_per_node && inh_idx < _node_num - 1)
		// 		// 				{
		// 		// 					inh_idx ++;
		// 		// 					inh_cnt = 0;
		// 		// 				}
		// 		// 				inh_cnt ++;
		// 		// 				_idx2node[t][i] = inh_idx;
		// 		// 				_idx2node[siter->type()][siter->id()] = inh_idx;
		// 		// 			}
		// 		// 		}
		// 		// 	}
		// 		// }

		// 		// fixed
		// 		vector<size_t> exc_cnts,inh_cnts;
		// 		exc_cnts.resize(_node_num,0);
		// 		inh_cnts.resize(_node_num,0);

		// 		size_t node_idx = 0;

		// 		for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
		// 			Type t = t_iter->first;
		// 			for (size_t i=0; i<t_iter->second->size(); i++) {
		// 				ID id(t,0,i);
		// 				exc_cnt = inh_cnt = 0;
		// 				for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
		// 					if(_synapses[siter->type()]->weight(siter->id()) >= 0)
		// 						exc_cnt ++;
		// 					else
		// 						inh_cnt ++;
		// 				}
		// 				size_t ni = 0;
		// 				if(node_idx < _node_num)
		// 				{
		// 					if(exc_cnts[node_idx] + exc_cnt > exc_per_node || inh_cnts[node_idx] + inh_cnt > inh_per_node)
		// 					{
		// 						node_idx ++;
		// 					}
		// 					ni = node_idx;
		// 				}

		// 				if(node_idx >= _node_num)
		// 				{
		// 					ni = 0;
		// 					size_t min = exc_cnts[0] + inh_cnts[0] + exc_cnt + inh_cnt;

		// 					for(size_t ii= 0; ii < _node_num; ii++)
		// 					{
		// 						if(exc_cnts[ii] + inh_cnts[ii] + exc_cnt + inh_cnt < min)
		// 						{
		// 							min = exc_cnts[ii] + inh_cnts[ii] + exc_cnt + inh_cnt;
		// 							ni = ii;
		// 						}
		// 					}
							
		// 				}

		// 				exc_cnts[ni] += exc_cnt;
		// 				inh_cnts[ni] += inh_cnt;

		// 				_idx2node[t][i] = ni;
		// 				for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
		// 					if(_synapses[siter->type()]->weight(siter->id()) >= 0)
		// 					{
		// 						_idx2node[siter->type()][siter->id()] = ni;
		// 					}
		// 					else
		// 					{
		// 						inh_cnt ++;
		// 						_idx2node[siter->type()][siter->id()] = ni;
		// 					}
		// 				}
						


		// 			}
		// 		}

		// 	}
		// 	printf("Split finished !\n");
		// 	break;
		case NSynB:
		{

			double theta =  0.00133;
			double lambda = 5.82;
			// printf("===NSynB:\n");
			size_t grp_size = 32;
			size_t grp_cur = 0;

			// size_t npart1 = _node_num/2;
			size_t npart1 = _node_num/2;
			size_t npart2 = 2;


			size_t exc_num = 0;
			size_t inh_num = 0;
			for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
				Type t = t_iter->first;
				for (size_t i=0; i<t_iter->second->size(); i++) {
					for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
						if(_synapses[siter->type()]->weight(siter->id()) >= 0)
							exc_num ++;
						else
							inh_num ++;
					}
				}
			}
			// printf("exc / inh : %ld  %ld \n",exc_num,inh_num);

			size_t exc_per_node = (exc_num -1)/npart1 + 1;
			size_t inh_per_node = (inh_num -1)/npart1 + 1;
			size_t exc_cnt = 0;
			size_t inh_cnt = 0;
			vector<size_t> exc_cnts(npart1,0),inh_cnts(npart1,0);
			vector<size_t> neu_num(npart1,0);
			size_t node_idx = 0;
			size_t ni = 0;
			for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
				Type t = t_iter->first;
				for (size_t i=0; i<t_iter->second->size(); i++) {
					grp_cur = (grp_cur + 1)% grp_size;
					exc_cnt = inh_cnt = 0;
					for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
						if(_synapses[siter->type()]->weight(siter->id()) >= 0)
							exc_cnt ++;
						else
							inh_cnt ++;
					}

					if(grp_cur == 0)
					{
						if(node_idx < npart1)
						{
							if((exc_cnts[node_idx] + exc_cnt > exc_per_node || inh_cnts[node_idx] + inh_cnt > inh_per_node))
								node_idx ++;
							ni = node_idx;
						}

						if(node_idx >= npart1)
						{
							ni = 0;
							double min = lambda * neu_num[0] + theta*(exc_cnts[0] + inh_cnts[0] + exc_cnt + inh_cnt);
							for(size_t ii= 1; ii < npart1; ii++)
							{
								double tmp = lambda * neu_num[ii] + theta*(exc_cnts[ii] + inh_cnts[ii] + exc_cnt + inh_cnt);
								if(tmp < min)
								{
									min = tmp;
									ni = ii;
								}
							}	
						}
					}

					exc_cnts[ni] += exc_cnt;
					inh_cnts[ni] += inh_cnt;
					_idx2node[t][i] = ni;
					neu_num[ni] ++;
					gettimeofday(&t3,NULL);
					for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
						_idx2node[siter->type()][siter->id()] = ni;
					}
					gettimeofday(&t4,NULL);
					timeno += (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
					
				}
			}
						
			node_idx = 0;
			size_t synapse_count = 0;
			if(npart2 > 1)
			{
				vector<bool> Flag(npart1,false);
				vector<size_t> part_size(npart1,0);
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++)
				{
					Type t = t_iter->first;
					for( size_t i=0; i<t_iter->second->size();i++)
					{
						node_idx = _idx2node[t][i];
						if(!Flag[node_idx])
						{
							Flag[node_idx] = true;
							// node_idx = _idx2node[t][i];
							part_size[node_idx] = (exc_cnts[node_idx] + inh_cnts[node_idx] -1)/npart2 + 1;
							//  syn cnt
							exc_cnts[node_idx] = 0;
							// grp_cur
							inh_cnts[node_idx] = 0;
						}
						

						_idx2node[t][i] = node_idx * npart2 + inh_cnts[node_idx];

						gettimeofday(&t3,NULL);
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) 
							_idx2node[siter->type()][siter->id()] = node_idx * npart2 + inh_cnts[node_idx];
						gettimeofday(&t4,NULL);
						timeno += (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
						
						exc_cnts[node_idx] += n2s_rev[t][i].size();
						if(exc_cnts[node_idx] >= part_size[node_idx])
						{
							inh_cnts[node_idx] ++;
							exc_cnts[node_idx] = 0;
						}
					}
				}
			}
			
		}
			// printf("Split finished !\n");
			break;
		default:
			{
				printf("===SYN_BASE\n");
				// int node_idx = 0;
				// size_t synapse_count = 0;
				// size_t synapse_per_node = _synapse_num/_node_num;
				// for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
				// 	Type t = t_iter->first;
				// 	for (size_t i=0; i<t_iter->second->size(); i++) {
				// 		ID id(t, 0, i);
				// 		_idx2node[t][i] = node_idx;
				// 		for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
				// 			_idx2node[siter->type()][siter->id()] = node_idx;
				// 		}
				// 		synapse_count += n2s_rev[t][i].size();
				// 		if (synapse_count >= (node_idx+1) * synapse_per_node && node_idx < _node_num - 1) {
				// 			node_idx++;	
				// 		}
				// 	}
				// }

				int node_idx = 0;
				size_t grp_size = 32;
				size_t grp_cur = 0;

				size_t npart1 = 16;
				size_t npart2 = 1;


				size_t synapse_count = 0;
				size_t synapse_per_node = _synapse_num/npart1;

				vector<size_t> part_size(npart1,0);

				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);
						_idx2node[t][i] = node_idx * npart2;
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx * npart2 ;
						}
						grp_cur = (grp_cur + 1)% grp_size;
						printf("%ld   %ld --------\n",grp_cur,node_idx);
						synapse_count += n2s_rev[t][i].size();
						part_size[node_idx] += n2s_rev[t][i].size();
						if ( (synapse_count >= (node_idx+1) * synapse_per_node && node_idx < _node_num - 1 ) && (grp_cur == 0)){
							node_idx++;	
						}
					}
				}

				node_idx = npart1 + 1;
				if(npart2 > 1)
				{
					for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++)
					{
						Type t = t_iter->first;
						for( size_t i=0; i<t_iter->second->size();i++)
						{
							ID id(t,0,i);
							if( node_idx != _idx2node[t][i])
							{
								node_idx = _idx2node[t][i];
								grp_size = (part_size[node_idx] -1)/npart2 + 1;
								grp_cur = 0;
								synapse_count = 0;
							}

							_idx2node[t][i] = node_idx * npart2 + grp_cur;
							for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
								_idx2node[siter->type()][siter->id()] = node_idx * npart1 + grp_cur;
							}

							synapse_count += n2s_rev[t][i].size();
							if(synapse_count >= grp_size)
							{
								grp_cur ++;
								synapse_count = 0;
							}
							
						}
					}
				}
				
			}
	}

	gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    printf("Split time:  %lf / %lf\n",timeuse,timeno);
	// exit(-1);

	print_mem("before clear n2s_rev");

	// for (auto iter = n2s_rev.begin(); iter != n2s_rev.end(); iter++) {
	// 	for (auto siter = iter->second.begin(); siter != iter->second.end(); siter++) {
	// 		siter->clear();
	// 		vector<ID> tmp;
	// 		siter->swap(tmp);
	// 	}
	// 	iter->second.clear();
	// 	vector<vector<ID>> tmp;
	// 	iter->second.swap(tmp);
	// }
	n2s_rev.clear();
	print_mem("after clear n2s_rev");

	return;
}

