
with open('test.txt', 'r') as f:
    lines = f.readlines()
    neuron_num = 0
    synapse_num = 0
    for line in lines:
        line = line.split(' ')
        neuron_num += int(line[0])
        synapse_num += int(line[1].split('\n')[0])
print(neuron_num, synapse_num)
