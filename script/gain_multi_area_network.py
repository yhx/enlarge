import json
import numpy as np
import sys


NEURON_FILE = '/archive/share/linhui2/downscale/neuron.log'
WEIGHT_FILE = '/archive/share/linhui2/downscale/weight_'
NEURON_RANGE_FILE = '/archive/share/linhui2/downscale/neuron_range_'
DATA_FILE = '/archive/share/linhui2/multi-area-model/multi-area-model/data/{label}/custom_Data_Model_{network_label}.json'

NEST_NETWORK_FILE = '/archive/share/linhui/new_bsim/bsim/nest_network/nest.network_0.2'
NEST_WEIGHT_FILE = '/archive/share/linhui/new_bsim/bsim/nest_network/nest.weight_0.2'
NEST_POISSON_FILE = '/archive/share/linhui/new_bsim/bsim/nest_network/nest.poisson_weight_0.2'

def read_data(label, network_label):
    '''
    Returns: group_num, neuron_num, w, w_sd, synapse_num, label_list, v_ext, w_ext

    group_num:   scalar, number of groups
    neuron_num:  [group_num] shape numpy array, neuron number of each group
    w:           [group_num, group_num] shape numpy array, mean synapse connection weight
                 w[target][source] : w from group[source] to target source
    w_sd:        same shape with w, synapse weight standard deviation
    synapse_num: [group_num, group_num] shape numpy array, number of synapses
    label_list:  [group_num] shape array, biological names of each group
    v_ext:       [group_num] shape array, background spike input rate for each group
    w_ext:       [group_num] shape array, background spike input weight for each group
    '''

    with open(DATA_FILE.format(label=label, network_label=network_label)) as f:
        data = json.load(f)
    area_list_data = data['area_list']
    population_list_data = data['population_list']
    structure_data = data['structure']
    with open(NEURON_FILE) as f:
        neuron_data = json.load(f)
    neuron_numbers_data = neuron_data #['neuron_numbers']
    synapses_data = data['synapses']
    synapse_weights_mean_data = data['synapse_weights_mean']
    synapse_weights_sd_data = data['synapse_weights_sd']
    distances_data = data['distances']

    area_list = []
    layer_list = []
    label_list = []

    group_num = 0
    for i in area_list_data:
        for j in structure_data[i]:
            area_list.append(i)
            layer_list.append(j)
            label_list.append(i + '-' + j)
            group_num += 1

    neuron_num = np.zeros(group_num)
    for i in range(group_num):
        neuron_num[i] = neuron_numbers_data[area_list[i]][layer_list[i]]

    w = np.zeros([group_num, group_num])
    w_sd = np.zeros_like(w)
    synapse_num = np.zeros_like(w)
    for target in range(group_num):
        for source in range(group_num):
            w[target][source] = synapse_weights_mean_data[area_list[target]][layer_list[target]][area_list[source]][layer_list[source]]
            w_sd[target][source] = synapse_weights_sd_data[area_list[target]][layer_list[target]][area_list[source]][layer_list[source]]
            synapse_num[target][source] = synapses_data[area_list[target]][layer_list[target]][area_list[source]][layer_list[source]]

    v_bg = 1e-2
    v_ext = np.zeros(group_num)
    w_ext = np.zeros(group_num)
    for i in range(group_num):
        v_ext[i] = v_bg * synapses_data[area_list[i]][layer_list[i]]['external']['external'] / neuron_num[i]
        w_ext[i] = synapse_weights_mean_data[area_list[i]][layer_list[i]]['external']['external']

    neuron_num = np.rint(neuron_num).astype(int)
    synapse_num = np.rint(synapse_num).astype(int)

    print('TotalNeuron:{:,}'.format(np.sum(neuron_num)))
    print('TotalSynapse:{:,}'.format(np.sum(synapse_num)))

    return group_num, neuron_num, w, w_sd, synapse_num, label_list, v_ext, w_ext

def read_poisson_weight(group_num, v_ext, w_ext):
    with open(NEST_POISSON_FILE, 'w') as f:
        for i in range(group_num):
            f.write(str(v_ext[i]) + ' ' + str(w_ext[i]) + '\n')

def read_weight(n, group_num, neuron_num, w, w_sd, synapse_num, label_list, v_ext, w_ext):
    '''
    用于根据权重生成每两个神经元群的连接权重
    '''
    for idx in range(n):  # 清空权重数组
        with open(NEST_WEIGHT_FILE, 'w') as f:
            f.write('')
        with open(NEST_NETWORK_FILE, 'w') as f:
            f.write('')

    # step1: 获取每个population的神经元id编码范围
    start_end_dict = {}  # 保存每一个population的start id和end id
    for idx in range(n):
        with open(NEURON_RANGE_FILE + str(idx) + '.log') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].replace('\n', '')
            area_num = int(lines[0])
            cur_line = 1
            for i in range(area_num):
                area_name = lines[cur_line]
                cur_line += 1

                pop_num = int(lines[cur_line])  # population数量
                cur_line += 1
                for j in range(cur_line, cur_line + pop_num * 2, 2):
                    pop_name = lines[j]
                    se = lines[j + 1].split(' ')
                    s = int(se[0])
                    e = int(se[1])
                    name = area_name + '-' + pop_name
                    if name not in start_end_dict:
                        start_end_dict[name] = {'start': s, 'end': e}
                cur_line += pop_num * 2
    label2id = {}
    i = 0
    for label in label_list: 
        label2id[label] = i  
        i += 1 
    # print(start_end_dict)

    # step2: 输出网络结构基本信息
    total_neuron_num = 0
    with open(NEST_NETWORK_FILE, 'w') as f:
        f.write(str(group_num) + '\n')  # population数量
        # 接下来一行是每个population中神经元的个数
        for i in range(group_num):
            f.write(str(neuron_num[i]) + ' ')
            total_neuron_num += neuron_num[i]
        f.write('\n')
    print('TOTAL NEURON NUMBER: ', total_neuron_num)

    # step3: 连接所有population的突触
    has_construct_weight = {}
    delay_num = np.zeros((1000, ))
    for idx in range(n):
        print('当前处理到:' + str(idx))
        with open(WEIGHT_FILE + str(idx) + '.log') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].replace('\n', '')
            # line_id = 0
            for line in lines:
                # line_id += 1
                # print(line_id)
                data = json.loads(line)
                source_area = data['source_area']
                target_area = data['target_area']
                source_population = data['source']
                target_population = data['target']
                synapses = data['synapse'] 
                s_label = source_area + '-' + source_population
                t_label = target_area + '-' + target_population
                s_id = label2id[s_label]
                t_id = label2id[t_label]

                if len(synapses) == 0: # TODO: 判断是否已经有了这部分权重?
                    continue

                # 输出网络结构
                with open(NEST_NETWORK_FILE, 'a+') as f3:
                    # 源神经元id 目的神经元id 源神经元数量 目的神经元数量
                    if str(s_id) + ' ' + str(t_id) not in has_construct_weight:
                        has_construct_weight[str(s_id) + ' ' + str(t_id)] = 1
                        f3.write(str(s_id) + ' ' + str(t_id) + ' ' + str(neuron_num[s_id]) + ' ' + str(neuron_num[t_id]) + '\n')

                s_start = start_end_dict[s_label]['start']  # 源population的神经元id范围
                t_start = start_end_dict[t_label]['start']  # 目的population的神经元id范围
                
                with open(NEST_WEIGHT_FILE, 'a+') as f2:
                    # 第一行：源神经元id 目的神经元id 源神经元数量 目的神经元数量 实际突触数量
                    f2.write(str(s_id) + ' ' + str(t_id)  + ' ' + str(neuron_num[s_id]) + ' ' + str(neuron_num[t_id]) +' ' + str(len(synapses)) + '\n')
                    for k in range(len(synapses)):
                        # 输出的每行数据格式是：
                        # 源神经元id 目的神经元id 连接权重 连接延迟 突触类型 
                        syn = synapses[str(k)]
                        delay_num[int(syn['delay'] / 0.1)] += 1
                        f2.write(str(int(syn['source'])-s_start) + ' ' + str(int(syn['target'])-t_start) + ' ' 
                            + str(syn['weight']) + ' ' + str(syn['delay']) + ' ' + str(syn['receptor']) + '\n')
    print('delay num:')
    for i in range(1000):
        print(str(delay_num[i]) + ' ', end="")
    print('')    

# e1e69d12090d6927ea450ea3072052b0
# 213d4830bc6b84ffbfbdca3b1dfd3bce
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(sys.argv[0] + 'need 3 params: label, network_label, process_number')
    else:
        group_num, neuron_num, w, w_sd, synapse_num, label_list, v_ext, w_ext = read_data(sys.argv[1], sys.argv[2])
        read_poisson_weight(group_num, v_ext, w_ext)
        # print(group_num, neuron_num, synapse_num)
        # print(v_ext.size, w_ext.size)
        read_weight(int(sys.argv[3]), group_num, neuron_num, w, w_sd, synapse_num, label_list, v_ext, w_ext)
