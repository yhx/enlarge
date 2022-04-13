import numpy as np
import sys

spikes = np.zeros((0,))
length = -1

for i in range(16):
    with open('./tmp/spike_nest_{0}.log'.format(i), 'r') as f:
        line = f.readlines()[0]
        line = line.split(' ')[:-1]
        line = list(map(int, line))
        line = np.array(line)
        length = max(length, line.max())
        spikes = np.concatenate((spikes, line))

with open('./tmp/neuron_gid.log', 'r') as f:
    line = f.readlines()[0]
    line = line.split(' ')[:-1]
    line = list(map(int, line))
    line = np.array(line)

line = line.astype(int)
idx = np.zeros((length+1,))
idx[line] = np.arange(len(line))
idx = idx.astype(int)

spikes = spikes.astype(int)
# print(length)
print("total spike number: {}".format(spikes.shape))
total_spikes = np.zeros((len(line),)).astype(int)
for i in range(len(spikes)):
    total_spikes[idx[spikes[i]]] += 1
print(total_spikes)

with open('./tmp/spike_count.log', 'w') as f:
    for i in range(len(total_spikes)):
        f.write(str(total_spikes[i]) + " ")

path = sys.argv[1]
with open(path, 'r') as f:
    line = f.readline().split(' ')[:-1]
    line = list(map(int, line))
line = np.sort(line)
total_spikes = np.sort(total_spikes)

print("total difference: {}".format(np.abs(line - total_spikes).sum()))
