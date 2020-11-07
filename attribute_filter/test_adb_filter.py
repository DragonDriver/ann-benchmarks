from __future__ import absolute_import
import numpy
import time
import random
import h5py
from adb import AnalyticDBAsync

def compute_recall(std, answer):
    hit_nums = 0.0
    for neighbor in answer:
        if neighbor in std:
            hit_nums += 1
    return hit_nums / len(answer)

dataset = 'sift-100000-1000',
client = AnalyticDBAsync(
    dataset=dataset,
    host='gp-bp1qswegsed309j28-master.gpdbmaster.rds.aliyuncs.com',
)
ds = h5py.File('../data/' + dataset + '.hdf5', 'r')
nums = []
selectivity = 0.3
field_fn = '../data/attribute_filter/' + dataset + '-%f' % (selectivity) + '-intfield.txt'
with open(field_fn, 'r') as f:
    nums = [int(line) for line in f.readline()]
train_size = len(ds['train'])
if not client.already_fit(train_size):
    if client.support_batch_fit():
        num_per_batch = 100000
        for i in range(0, train_size, num_per_batch):
            end = min(i + num_per_batch, train_size)
            train = numpy.array(ds['train'][i:end])
            client.batch_fit(train, nums[i:end], train_size)
    else:
        train = numpy.array(ds['train'])
        client.fit(train_size, nums)

qs = numpy.array(ds['test'])
upper_bound = round(train_size - train_size * selectivity)
topk = 50
client.batch_query(qs, topk, upper_bound)
ans = client.get_batch_results()
std_fn = '../data/attribute_filter/' + dataset + '-%f' % (selectivity) + '-std.txt'
stds = []
with open(std_fn, 'r') as f:
    import ast
    stds = [list(ast.literal_eval(line)) for line in f.readlines()]
recall = 0.0
for i in range(len(ans)):
    recall += compute_recall(stds[i], ans[i])
print("average recall: ", recall / len(ans))

client.done()
ds.close()
