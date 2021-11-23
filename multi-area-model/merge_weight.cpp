#include <cstdlib>
#include <time.h>
#include <fstream>
#include <vector>

#include "../include/BSim.h"

using namespace std;

// 存储网络权重的文件
const char *FILE_WEIGHT = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.weight_0.2_0.117";
// 存储经过merge后的程序
const char *FILE_WEIGHT_MERGE = "/archive/share/linhui/new_bsim/bsim/nest_network/nest.weight_merge_0.2_0.117";

struct SynInfo {
    int s_pop_id, t_pop_id, receptor;
    real w, d;

    SynInfo(int s_pop_id, int t_pop_id, real w, real d, int receptor) {
        this->s_pop_id = s_pop_id;
        this->t_pop_id = t_pop_id;
        this->w = w;
        this->d = d;
        this->receptor = receptor;
    }
};

void merge_network() {
    ifstream f_weight(FILE_WEIGHT, ios::in);
    ofstream f_weight_merge(FILE_WEIGHT_MERGE, ios::out);

    // 源population的id，目的population的id，源pop中神经元数量，目的pop中神经元数量，突触数量
    int s_id, t_id, s_num, t_num, syn_num;
    // 源poppulation中id编号，目的population中id编号，突触类型
    int s_pop_id, t_pop_id, receptor;
    real w, d;
    int pop_num = 254;              
    int synapse_num[300][300] = {};  // 记录每一个突触连接的连接数
    vector<vector<SynInfo> > synapse_info(pop_num * pop_num); 
    while(f_weight >> s_id >> t_id >> s_num >> t_num >> syn_num) {
        // cout << "syn_num: " << syn_num << endl;
        synapse_num[s_id][t_id] += syn_num;
        for (int i = 0; i < syn_num; ++i) {  
            f_weight >> s_pop_id >> t_pop_id >> w >> d >> receptor;
            synapse_info[s_id * pop_num + t_id].push_back(SynInfo(s_pop_id, t_pop_id, w, d, receptor));
        }
    }
    // cout << "max_delay: " << max_delay << endl;
    f_weight.close();

    for (int i = 0; i < pop_num; ++i) {
        for (int j = 0; j < pop_num; ++j) {
            f_weight_merge << i << " " << j << " " << synapse_num[i][j] << endl;
            for (int k = 0; k < synapse_num[i][j]; ++k) {
                f_weight_merge << synapse_info[i * pop_num + j][k].s_pop_id << " " 
                    << synapse_info[i * pop_num + j][k].t_pop_id << " "
                    << synapse_info[i * pop_num + j][k].w << " "
                    << synapse_info[i * pop_num + j][k].d << " "
                    << synapse_info[i * pop_num + j][k].receptor << endl; 
            }
        }
    }
    f_weight_merge.close();
}

int main(int argc, char **argv) {
    merge_network();
}