from __future__ import absolute_import
import numpy
import time
import random
from vearch_320 import VearchIVFFLAT

def compute_recall(std, answer):
    hit_nums = 0.0
    for neighbor in answer:
        if neighbor in std:
            hit_nums += 1
    return hit_nums / len(answer)

def gen_file_name(idx):
    s = "%05d" % idx
    return "/mnt/sift1b/binary_" + "128d_" + s + ".npy"

ncentroids = 8192
client = VearchIVFFLAT(ncentroids)
for i in range(0, 1000):
    print("file num: " + str(i))
    fname = gen_file_name(i)
    vectors = numpy.load(fname)
    fnum = open('/mnt/intvalues/num_' + str(i) + '.txt', 'r')
    n1 = fnum.readlines()[0].split()
    nums = []
    for j in range(len(n1)):
        nums.append(int(n1[j]))
    fnum.close()
    if i == 0:
        client.fit(vectors, nums, 0, i)
    else:
        client.fit(vectors, nums, 1, i)

client._create_index()


# query
data_q = numpy.load("/mnt/sift1b/query.npy")
topk = 50
qs = data_q[0:1000] 
nprobe = 16
upper = 9999
client.set_query_arguments(nprobe)
t = time.time()
client.batch_query(qs, topk, upper)
print("Time spent " + str(time.time() - t))
ids = client.get_batch_results()
# print(ids)

ground = open('/mnt/top_50_select_9.txt', 'r')
ground_list = []
lines = ground.readlines()

for l in lines:
    line = l.strip().split(',')
    for u in range(len(line)):
        line[u] = int(line[u])
    ground_list.append(line)

sum_radio = 0.0
for index, item in enumerate(ids):
    tmp = set(ground_list[index]).intersection(set(item))
    sum_radio = sum_radio + len(tmp) / len(item)
print(round(sum_radio / len(ids), 3))
