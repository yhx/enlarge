
#include <float.h>

#include "../utils/utils.h"
#include "Network.h"

void Network::splitNetwork(SplitType split, const char *name)
{

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

	if (_node_num <= 1) {
		return;
	}

	switch (split) {
		case NeuronBalance:
			{
				printf("===NEU_BASE\n");
				int node_idx = 0;
				size_t neuron_count = 0;
				size_t neuron_per_node = _neuron_num/_node_num;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);
						_idx2node[t][i] = node_idx;
						neuron_count += 1;
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}
						if (neuron_count>= (node_idx+1) * neuron_per_node && node_idx < _node_num - 1) {
							node_idx++;	
						}
					}
				}
			}
			break;
		case RoundRobin:
			{
				printf("===ROUND_ROBIN\n");
				size_t neuron_count = 0;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);
						int node_idx = neuron_count % _node_num;
						_idx2node[t][i] = node_idx;
						neuron_count += 1;
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}
					}
				}
			}
			break;
		case GrpRR:
			{
				printf("Group RoundRobin\n");
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
				printf("Synapse Bestfit\n");
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
					string s2 = s + ".graph";
					printf("===Metis 0\n");
					save_graph(s2.c_str());
					exit(0);
				} else {
					printf("===Metis\n");
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
				printf("===BestFit\n");
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
			}
			break;
		default:
			{
				printf("===SYN_BASE\n");
				int node_idx = 0;
				size_t synapse_count = 0;
				size_t synapse_per_node = _synapse_num/_node_num;
				for (auto t_iter = _neurons.begin(); t_iter != _neurons.end(); t_iter++) {
					Type t = t_iter->first;
					for (size_t i=0; i<t_iter->second->size(); i++) {
						ID id(t, 0, i);
						_idx2node[t][i] = node_idx;
						for (auto siter = n2s_rev[t][i].begin(); siter != n2s_rev[t][i].end(); siter++) {
							_idx2node[siter->type()][siter->id()] = node_idx;
						}
						synapse_count += n2s_rev[t][i].size();
						if (synapse_count >= (node_idx+1) * synapse_per_node && node_idx < _node_num - 1) {
							node_idx++;	
						}
					}
				}
			}
	}

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

