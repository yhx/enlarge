
with open('sim.mpi_0.log', 'r') as f:
    lines = f.readlines()

n = 2
depth = 3

spikes_time = []
for i in range(n * depth):
    spikes_time.append([])
for i in range(len(lines)):
    idxs = lines[i].split(' ')
    for idx in idxs:
        if idx == '\n':
            continue
        spikes_time[int(idx)].append(i)

# 输出spike时间
with open('./spike_time.bsim.log', 'w') as f:
    f.write('')
with open('./spike_time.bsim.log', 'w+') as f:
    for i in range(len(spikes_time)):
        for j in range(len(spikes_time[i])):
            f.write(str(int(spikes_time[i][j])) + ' ')
        f.write('\n')
