
print("Node", end="")
for i in range(1, 9):
    print(" & " + str(i), end="")
print(" \\\\ \\hline")

print("Forward", end="")
for i in range(1, 9):
    print(" & \\makecell{ $N=" + str(1000*300*i) + "$ \\\\ $S=" + str( 1000 * 1000 * (i * 300 - 1 + i * 300 // 4) ) + "$ }", end="")
print(" \\\\ \\hline")

neuron_nums = [15000, 21213, 25981, 30000, 33541, 36743, 39687, 42427]
print("Circle", end="")
for i in range(1, 9):
    print(" & \\makecell{ $N=" + str(neuron_nums[i-1]*3) + "$ \\\\ $S=" + str( (neuron_nums[i-1] ** 2) * 3 ) + "$ }", end="")
print(" \\\\ \\hline")

neuron_nums = [1000, 1195, 1267, 1290, 1290, 1279, 1261, 1240]
pop_nums = [6, 7, 8, 9, 10, 11, 12, 13]
print("FC", end="")
for i in range(1, 9):
    print(" & \\makecell{ $N=" + str(neuron_nums[i-1]*pop_nums[i-1]) + "$ \\\\ $S=" + str( (neuron_nums[i-1] ** 2) * pop_nums[i-1] * (pop_nums[i-1]-1) )  + "$ }", end="")
print(" \\\\ \\hline")
