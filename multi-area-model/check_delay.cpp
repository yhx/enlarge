#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <vector>
#include "../include/BSim.h"

using namespace std;

const char *FILE_NETWORK = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.network_0.30_0.16";         // 存储网络结构的文件
const char *FILE_WEIGHT = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.weight_merge_0.30_0.16";
const char *FILE_POISSON = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.poisson_weight_0.30_0.16";  // 存储poisson分布的文件

// const char *FILE_NETWORK = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.network_0.20_0.117";         // 存储网络结构的文件
// const char *FILE_WEIGHT = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.weight_merge_0.20_0.117";
// const char *FILE_POISSON = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.poisson_weight_0.20_0.117";  // 存储poisson分布的文件

void check_delay() {
    ifstream f_weight(FILE_WEIGHT, ios::in);
    // 源population的id，目的population的id，源pop中神经元数量，目的pop中神经元数量，突触数量
    int s_id, t_id, s_num, t_num, syn_num;
    // 源poppulation中id编号，目的population中id编号，突触类型
    int s_pop_id, t_pop_id, receptor;
    // 权重，延迟
    real w;
    real d;
    int delay[1000] = {};

    while(f_weight >> s_id >> t_id >> syn_num) {

        if (syn_num == 0) {
            continue;
        }

        for (int i = 0; i < syn_num; ++i) {  
            f_weight >> s_pop_id >> t_pop_id >> w >> d >> receptor;
            // cout << d / 0.1 << endl;
            delay[int(d / 0.1)] += 1;
        }
    }
    // cout << "max_delay: " << max_delay << endl;
    f_weight.close();
    for (int i = 0; i < 1000; ++i) {
        cout << delay[i] << " ";
    }
}

void build_population() {
    ifstream f_net(FILE_NETWORK, ios::in);
    int r_data;
    f_net >> r_data;
    const int pop_num = r_data;  // population总数
    cout << "pop_num: " << pop_num << endl;
    int pop_neuron_num[pop_num+1];  // 每个population的神经元数量
    for (int i = 0; i < pop_num; ++i) {
        f_net >> pop_neuron_num[i];
    }
    cout << endl;
    cout << "population neuron number:" << endl;
    for (int i = 0; i < pop_num; ++i) {
        cout << pop_neuron_num[i] << " ";
    }
    cout << endl;

    ifstream f_weight(FILE_WEIGHT, ios::in);
    // 源population的id，目的population的id，源pop中神经元数量，目的pop中神经元数量，突触数量
    int s_id, t_id, s_num, t_num, syn_num;
    // 源poppulation中id编号，目的population中id编号，突触类型
    int s_pop_id, t_pop_id, receptor;
    // 权重，延迟
    real w;
    real d;
    vector<double> res;

    while(f_weight >> s_id >> t_id >> syn_num) {
        for (int i = 0; i < syn_num; ++i) {  
            f_weight >> s_pop_id >> t_pop_id >> w >> d >> receptor;
        }
        res.push_back((double)(syn_num)/(double)(pop_neuron_num[s_id] * pop_neuron_num[t_id]));
    }
    cout << "connection probability: " << endl;
    f_weight.close();
    for (int i = 0; i < res.size(); ++i) {
        cout << res[i] << " ";
    }
    cout << endl;
}

int main(int argc, char **argv) {
    // check_delay();
    build_population();
    return 0;
}
