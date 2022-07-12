import numpy as np
num = np.zeros((3,))
with open('test.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        i = 0
        for nu in line:
            nu = nu.split('\n')
            num[i] += int(nu[0])
            i += 1
print(num)