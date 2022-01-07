
print("\\multicolumn{2}{c}{Node Number}", end="")
for i in range(1, 9):
    print(" & " + str(i), end="")
print(" \\\\ \\hline")

print("\\multirow{2}*{Forward} & N", end="")
for i in range(1, 9):
     print(" & $" + str(1000*300*i) + "$", end="")
print(" \\\\ \n\\cline{2-10}\n ~ & S", end="")
for i in range(1, 9):
     print(" & $" + str( 1000 * 1000 * (i * 300 - 1 + i * 300 // 4)) + "$", end="")
print(" \\\\ \n\\hline")

neuron_nums = [15000, 21213, 25981, 30000, 33541, 36743, 39687, 42427]
print("\\multirow{2}*{Circle} & N", end="")
for i in range(1, 9):
    print(" & $" + str(neuron_nums[i-1]*3) + "$", end="")
print(" \\\\ \n\\cline{2-10}\n  ~ & S", end="")
for i in range(1, 9):
    print(" & $" + str( (neuron_nums[i-1] ** 2) * 3 ) + "$", end="")
print(" \\\\ \n\\hline")

neuron_nums = [1000, 1195, 1267, 1290, 1290, 1279, 1261, 1240]
pop_nums = [6, 7, 8, 9, 10, 11, 12, 13]

print("\\multirow{2}*{FC} & N", end="")
for i in range(1, 9):
    print(" & $" + str(neuron_nums[i-1]*pop_nums[i-1]) + "$", end="")
print(" \\\\ \n\\cline{2-10}\n ~ &  S", end="")
for i in range(1, 9):
    print(" & $" + str( (neuron_nums[i-1] ** 2) * pop_nums[i-1] * (pop_nums[i-1]-1) )  + "$", end="")
print(" \\\\ \n\\hline")
