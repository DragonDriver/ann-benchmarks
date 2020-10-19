import os
import logging
import pdb
import time
import random
from multiprocessing import Process
import numpy as np
import sklearn.preprocessing
from milvus import DataType, Milvus

SIFT_VECTORS_PER_FILE = 100000
MAX_NQ = 10001
FILE_PREFIX = "binary_"
SIFT_SRC_DATA_DIR = '/home/sift1b/'
SIFT_SRC_GROUNDTRUTH_DATA_DIR = SIFT_SRC_DATA_DIR + 'gnd'

_HOST = '127.0.0.1'
_PORT = '19530'  # default value

def gen_file_name(idx):
    s = "%05d" % idx
    return SIFT_SRC_DATA_DIR + FILE_PREFIX + "128d_" + s + ".npy"

def get_vectors_from_binary(nq):
    file_name = SIFT_SRC_DATA_DIR + "query.npy"
    data = np.load(file_name)
    return data[0:nq].tolist()

vector_type = DataType.FLOAT_VECTOR

def normalize(X):
    X = X.astype(np.float32)

def gen_partition_name(num):
    s = "partition_" + str(num)
    return s

milvus = Milvus(_HOST, _PORT)
vectors_per_file = SIFT_VECTORS_PER_FILE
collection_name = 'test_collection'


# create collection
collection_param = {
    "fields": [
        {"name": "int_field", "type": DataType.INT32},
        {"name": "vec_float", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128, "metric_type": "L2"}}
    ],
    "segment_row_limit": 4000000,
    "auto_id": False
}
print("create collection...")
milvus.create_collection(collection_name, collection_param)
milvus.compact(collection_name)
print(milvus.get_collection_info("test_collection"))

tag = []
for i in range(25):
    s = gen_partition_name(i)
    tag.append(s)
    milvus.create_partition(collection_name, s)


# get data from first 1000 files to get 100m rows
for i in range(25):
    for j in range(40):
        file_name = gen_file_name(j + 40 * i)
        data = np.load(file_name)
        vectors = data[:].tolist()
        start_id = (j + 40 * i) * vectors_per_file
        end_id = start_id + len(vectors)
        print("Start id: %s, end id: %s" % (start_id, end_id))
        ids = [k for k in range(start_id, end_id)]
        # int field values
        int_values = []
        fnum = open('../intvalues/num_' + str(i * 40 + j) + '.txt', 'r')
        n1 = fnum.readlines()[0].split()
        for u in range(len(n1)):
            int_values.append(int(n1[u]))
        fnum.close()
        hybrid_entities = [
            {"name": "int_field", "values": int_values, "type": DataType.INT32},
            {"name": "vec_float", "values": vectors, "type": DataType.FLOAT_VECTOR}
        ]
        res_ids = milvus.insert(collection_name, hybrid_entities, ids=ids, partition_tag=tag[i])
        assert ids == res_ids
        milvus.flush([collection_name])
        res = milvus.count_entities(collection_name)
        print("Row count: " + str(res))


def get_recall_value(true_ids, result_ids):
    """
    Use the intersection length
    """
    sum_radio = 0.0
    for index, item in enumerate(result_ids):
        # tmp = set(item).intersection(set(flat_id_list[index]))
        tmp = set(true_ids[index]).intersection(set(item))
        sum_radio = sum_radio + len(tmp) / len(item)
        # logger.debug(sum_radio)
    return round(sum_radio / len(result_ids), 3)

def get_ids(result):
    ids = []
    for item in result:
        ids.append([entity.id for entity in item])
    return ids


print("Create index ......")
milvus.create_index(collection_name, "vec_float", {"index_type": "IVF_FLAT", "params": {"nlist": 8192}, "metric_type": "L2"})
print("Create index done.")
print()


def get_groundtruth_ids():
    fname = "idx_1000M.ivecs"
    fname = SIFT_SRC_GROUNDTRUTH_DATA_DIR + "/" + fname
    a = np.fromfile(fname, dtype = 'int32')
    d = a[0]
    true_ids = a.reshape(-1, d + 1)[:, 1:].copy()
    return true_ids

'''
# perform query
nq = 1000
query_vectors = get_vectors_from_binary(nq)
topk = 50
res_ids = [0, 1, 2, 3]
passed = False

# experiment
print("==========Experiment ==========")
for strategy in [2, 2, 3]:
    if passed:
        print("==========Strategy " + str(strategy) + "==========")
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "range": {
                        "int_field": {"GT": -1, "LT": 100}
                    }
                },
                {
                    "vector": {
                        "vec_float": {
                            "topk": topk, "query": query_vectors, "params": {"nprobe": 32}
                        }
                    }
                }
            ],
        },
        "strategy": strategy,
        "delta": 1.01
    }

    if passed:
        print("Start search ..")
    t0 = time.time()
    results = milvus.search(collection_name, query_hybrid, partition_tags=tag[0:10])
    if passed:
        print("Time spent for search: " + str(time.time() - t0))
        #res_ids[strategy] = get_ids(results)
    passed = True


# check recall rate
#print("Strategy 2 recall rate: " + str(get_recall_value(res_ids[1], res_ids[2])))
#print("Strategy 3 recall rate: " + str(get_recall_value(res_ids[1], res_ids[3])))