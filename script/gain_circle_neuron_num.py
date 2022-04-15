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

N = 1000
for node_num in range(1, 9):
    print(math.floor(math.sqrt(1000 ** 2 * 15 * node_num / ((node_num + 5) * (node_num + 4) / 2))), end=' ')
print('')
