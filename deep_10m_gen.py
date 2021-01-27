#!/usr/bin/env python

import os
import struct
import numpy as np
import time
import datetime
import h5py
import sys

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def bvecs_to_ndarray(bvecs_fn, size=sys.maxsize):
    with open(bvecs_fn, 'rb') as f:
        fsize = os.path.getsize(bvecs_fn)
        dimension, = struct.unpack('i', f.read(4))
        print('dimension: ', dimension)
        vector_nums = fsize // (4 + dimension)
        vector_nums = min(vector_nums, size)
        print('vector_nums: ', vector_nums)

        v = np.zeros((vector_nums, dimension))
        f.seek(0)
        begin = time.time()
        for i in range(vector_nums):
            if i % 10000 == 0:
                print("[{}] handle {}th vector, time cost: {}".format(datetime.datetime.now(), i, time.time() - begin))
            f.read(4)
            v[i] = struct.unpack('B' * dimension, f.read(dimension))

        return v

def handle_deep_1b(out_fn, train_num, query_num, distance, count=100):
    import sklearn.preprocessing
    f = h5py.File(out_fn, 'w')
    f.attrs['distance'] = distance
    f.attrs['point_type'] = 'float'
    test = fvecs_read('/mnt/cifs/milvus_paper/deep1b/deep1B_queries.fvecs')
    query_num = min(query_num, len(test))
    test = test[:query_num]
    if distance == 'euclidean':
        test = sklearn.preprocessing.normalize(test, axis=1, norm='l2')
    dimension = len(test[0])
    f.create_dataset(
        'test',
        (len(test), dimension),
        dtype=test.dtype,
    )[:] = test

    # no enough memory to directly use fvecs_read
    train_fn = '/mnt/cifs/milvus_paper/deep1b/base/base.fvecs'
    if train_num == 10000000 and False:
        # no necessary to use standard sample dataset
        train_fn = '/mnt/cifs/milvus_paper/deep1b/deep10M.fvecs'
    with open(train_fn, 'rb') as ftrain:
        train_size = os.path.getsize(train_fn)
        train_dimension, = struct.unpack('i', ftrain.read(4))
        print("train_dimension: ", train_dimension)
        assert train_dimension == dimension
        vector_nums = train_size // (4 + 4 * train_dimension)
        vector_nums = min(vector_nums, train_num)
        print("vector_nums: ", vector_nums)
        train = f.create_dataset(
            'train',
            (vector_nums, dimension),
            dtype=test.dtype,
        )
        begin = time.time()
        ftrain.seek(0)
        for i in range(vector_nums):
            ftrain.read(4) # ignore dimension, faster
            train[i] = struct.unpack('f' * train_dimension, ftrain.read(train_dimension * 4))
            if distance == 'euclidean':
                train[i] = sklearn.preprocessing.normalize([train[i]], axis=1, norm='l2')[0]
            if i % 100000 == 0:
                print("[{}] handle {}th vector, time cost: {}".format(datetime.datetime.now(), i, time.time() - begin))

    neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    distances = f.create_dataset('distances', (len(test), count), dtype='f')
    from ann_benchmarks import datasets
    from ann_benchmarks.algorithms.bruteforce_angular import BruteForceBLAS
    bf = BruteForceBLAS(distance, precision=train.dtype)
    train = datasets.dataset_transform[distance](train)
    test = datasets.dataset_transform[distance](test)
    bf.fit(np.array(train))
    for i, x in enumerate(test):
        if i % 1000 == 0:
            print('[{}] {}/{}...'.format(datetime.datetime.now(), i, len(test)))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    f.close()

def main():
    # for test
    handle_deep_1b('/home/ljq/work/ann-benchmarks/datagen/deep/deep-100000-1000-euclidean.hdf5', 100000, 1000, 'euclidean')
    handle_deep_1b('/home/ljq/work/ann-benchmarks/datagen/deep/deep-100000-1000-angular.hdf5', 100000, 1000, 'angular')
    # euclidean_out_fn = '/cifs/data/milvus_paper/deep1b/deep-10m-euclidean.hdf5'
    # handle_deep_1b(euclidean_out_fn, 10000000, 10000, 'euclidean')
    angular_out_fn = '/home/ljq/work/ann-benchmarks/datagen/deep/deep-10m-angular.hdf5'
    handle_deep_1b(angular_out_fn, 10000000, 10000, 'angular')

if __name__ == "__main__":
    main()
