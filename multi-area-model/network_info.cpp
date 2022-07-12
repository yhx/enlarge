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

int main(int argc, char **argv) {

	ifstream f_net(FILE_NETWORK, ios::in);
	int r_data;
	f_net >> r_data;
	const int pop_num = r_data;  // population总数
	cout << "pop_num: " << pop_num << endl;
	int pop_neuron_num[pop_num+1];  // 每个population的神经元数量
	int total_neuron_num = 0;
	for (int i = 0; i < pop_num; ++i) {
		f_net >> pop_neuron_num[i];
		cout << pop_neuron_num[i] << " ";
		total_neuron_num += pop_neuron_num[i];
	}
	cout << endl;
	cout << "total_neuron_num: " << total_neuron_num << endl;
    
    return 0;
}
