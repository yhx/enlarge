#include <stdlib.h>
#include <time.h>
#include <fstream>
#include "../include/BSim.h"

using namespace std;

const char *FILE_NETWORK = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.network_0.20_0.117";         // 存储网络结构的文件
const char *FILE_WEIGHT = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.weight_merge_0.20_0.117";
const char *FILE_POISSON = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.poisson_weight_0.20_0.117";  // 存储poisson分布的文件

// const char *FILE_NETWORK = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.network_0.01";         // 存储网络结构的文件
// const char *FILE_WEIGHT = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.weight_merge_0.01";
// const char *FILE_POISSON = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.poisson_weight_0.01";  // 存储poisson分布的文件

const real dt=1e-4;

void build_population(Network &net, Population **pop, const int pop_num, const int *pop_neuron_num) {
    cout << "TOTAL POPULATION NUMBER: " << pop_num << endl;
    for (int i = 0; i < pop_num; ++i) {
        pop[i] = net.createPopulation(i, pop_neuron_num[i], IAFNeuron(dt));     
        print_mem(("BUILD NEURON: After build " + to_string(i) + "-th population").c_str());
    }
}

void connect_network(Network &net, Population **pop) {
    ifstream f_weight(FILE_WEIGHT, ios::in);
    // 源population的id，目的population的id，源pop中神经元数量，目的pop中神经元数量，突触数量
    int s_id, t_id, s_num, t_num, syn_num;
    // 源poppulation中id编号，目的population中id编号，突触类型
    int s_pop_id, t_pop_id, receptor;
    // 权重，延迟
    real w, d;
    // real max_delay = 0;
    // int test = 0;
    // while(f_weight >> s_id >> t_id >> s_num >> t_num >> syn_num) {
    while(f_weight >> s_id >> t_id >> syn_num) {
        // cout << "syn_num: " << syn_num << endl;
        if (syn_num == 0) {
            continue;
        }
        // test++;
        // if (test >= 2) {
        //     break;
        // }
        Population *p_src = pop[s_id];                  // 源population
        Population *p_dst = pop[t_id];                  // 目的population
        int *idx_src = new int[syn_num+1];              // 源神经元id数组
        int *idx_dst = new int[syn_num+1];              // 目的神经元id数组
        real *weight = new real[syn_num+1];             // 权重数组
        real *delay = new real[syn_num+1];              // 延迟数组
        real *tau = new real[syn_num+1];                // 突触参数
        SpikeType *sp = new SpikeType[syn_num+1];       // 突触类型
        for (int i = 0; i < syn_num; ++i) {  
            f_weight >> s_pop_id >> t_pop_id >> w >> d >> receptor;
            idx_src[i] = s_pop_id;
            idx_dst[i] = t_pop_id;
            weight[i] = w;
            delay[i] = d * 1e-3;  // change unit ms -> s
            if (weight[i] >= 0) {
                sp[i] = Exc;  
            } else {
                sp[i] = Inh;
            }
            // max_delay = max(max_delay, d);
        }
        // 将两个population连在一起
        net.connect(p_src, p_dst, idx_src, idx_dst, syn_num, weight, delay, sp);

        delArray(idx_src);
        delArray(idx_dst);
        delArray(weight);
        delArray(delay);
        delArray(sp);
        print_mem(("BUILD SYNAPSE: After connect " + to_string(s_id) + " and " + to_string(t_id) + " populations").c_str());
    }
    // cout << "max_delay: " << max_delay << endl;
    f_weight.close();
}

void connect_poisson_generator(Network &net, Population **pop, const int pop_num, const int *pop_neuron_num) {
    ifstream f_poisson(FILE_POISSON, ios::in);
    real poisson_mean, poisson_weight;
    string pop_name;
    for (int i = 0; i < pop_num; ++i) {
        // if (i >= 3) {
        //     break;
        // }
        f_poisson >> pop_name >> poisson_mean >> poisson_weight;
        // should compute poisson mean in each dt! so, it should time dt.
        real *poisson_means = getConstArray((real)(poisson_mean * dt), pop_neuron_num[i]);
        // real *poisson_means = getConstArray((real)(poisson_mean / 30000), pop_neuron_num[i]);
        real *poisson_weights = getConstArray((real)(poisson_weight), pop_neuron_num[i]);
        real *delay = getConstArray(dt, pop_neuron_num[i]); 
        net.connect_poisson_generator(pop[i], poisson_means, poisson_weights, delay, NULL);
        delArray(poisson_means);
        delArray(poisson_weights);
        delArray(delay);
        print_mem(("BUILD POISSON: After connect " + to_string(i) + "-th poisson").c_str());
    }
    f_poisson.close();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int node_id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    Network net(dt);
    cout << "node id: " << node_id << endl;

    if (node_id == 0) {
        ifstream f_net(FILE_NETWORK, ios::in);
        int r_data;
        f_net >> r_data;
        const int pop_num = r_data;  // population总数
        cout << "pop_num: " << pop_num << endl;
        int pop_neuron_num[pop_num+1];  // 每个population的神经元数量
        for (int i = 0; i < pop_num; ++i) {
            f_net >> pop_neuron_num[i];
        }

        // create neuron populations
        cout << "START CREATE POPULATION!" << endl;
        Population *pop[pop_num + 1];
        build_population(net, pop, pop_num, pop_neuron_num);
        cout << "FINISH CREATE POPULATION!" << endl;
        
        // connect populations according to record data
        cout << "START CONNECT NETWORK!" << endl;
        connect_network(net, pop);
        cout << "FINISH CONNECT NETWORK!" << endl;

        // connect poisson generator according to record data
        connect_poisson_generator(net, pop, pop_num, pop_neuron_num);

        f_net.close();
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    const real run_time = 0.1;
    MNSim mn(&net, dt);

    if (node_id == 0) {
        int parts = 16;
        SplitType split = SynapseBalance;
        const char * name = "multi_area_model_20_117";
        mn.build_net(parts, split, name);
        mn.save_net(name);
    }
    // mn.run(run_time, 1);
    
    return 0;
}
