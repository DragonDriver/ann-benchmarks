#!/usr/bin/env python

import milvus
import numpy
import h5py
import datetime
import time
from ann_benchmarks.algorithms.milvus_ivf_flat import MilvusIVFFLAT
from ann_benchmarks.algorithms.milvus_hnsw import MilvusHNSW

def ivfflat_warmup(metric='euclidean',
                   dataset='sift-1b-euclidean',
                   index_type='IVF_FLAT',
                   nlist=2048,
                   nprobe=10):
    print('ivfflat warmup')
    client = MilvusIVFFLAT(metric, dataset, index_type, nlist)
    ds = h5py.File('data/' + dataset + '.hdf5', 'r')
    tests = numpy.array(ds['test'])
    query = tests[0]
    client.set_query_arguments(nprobe)
    status, result = client.query(query, 50).result()
    print("status: ", status)
    print("result: ", result)
    print("len(result): ", len(result))

def hnsw_warmup(metric='euclidean',
                dataset='sift-1b-euclidean',
                method_param={"M": 4, "efConstruction": 500},
                efSearch=10):
    print('hnsw warmup')
    client = MilvusIVFFLAT(metric, dataset, method_param)
    ds = h5py.File('data/' + dataset + '.hdf5', 'r')
    tests = numpy.array(ds['test'])
    query = tests[0]
    client.set_query_arguments(efSearch)
    status, result = client.query(query, 50).result()
    print("result: ", result)
    print("len(result): ", len(result))

if __name__ == "__main__":
    print('[{}] ivfflat begin warmup'.format(datetime.datetime.now()))
    begin = time.time()
    ivfflat_warmup()
    print('[{}] ivfflat warmup done'.format(datetime.datetime.now()))
    print("ivfflat warmup cost: ", time.time() - begin)

