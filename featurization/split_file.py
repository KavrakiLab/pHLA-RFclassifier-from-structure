import math
import sys

confs = []
f = open(sys.argv[1], 'r')
for line in f:
  confs.append(line)
f.close()

print(len(confs))
num_confs_per_file = int(math.ceil(len(confs) / 100.0))

confs_index = 0
for i in range(100):
  f = open("confs" + str(i) + ".txt", 'w')
  for j in range(num_confs_per_file):
    f.write(confs[confs_index])
    confs_index += 1
    if confs_index == len(confs): break
  f.close()
