# 计算pattern_circle的weak scaling中每步的神经元数量
import math

node_num = 15000
for n in range(1, 9):
    print(math.floor((-1 + math.sqrt(1 + 4 * n * (node_num + node_num**2))) / 2), end=' ')
print('')

N = 1500
D = 300
for i in range(1, 9):
    print(math.floor( ( ( N**2 * (D - 1) + N * D) * i + N**2 ) / (N**2 + N) ), end=' ')
print('')

# N = 1000
# P = 6
N = 2000
p = 10
b_0 = N**2 * p * (p - 1) + N * p
for node_num in range(1, 9):
    p = node_num + 10
    neuron_num = math.floor( (-p + math.sqrt(p**2 + 4 * node_num * b_0 * p * (p - 1))) / (2 * p * (p - 1)) )
    print(neuron_num, end=' ')
    # print(neuron_num * neuron_num * (p * (p - 1) / 2) + neuron_num * p)
print('')
