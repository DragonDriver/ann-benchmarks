#!/usr/bin/env python

import numpy
import struct
import os
import random

num = 20
fn = "/mnt/cifs/milvus_paper/deep1b/deep10M.fvecs"
fn = "/mnt/cifs/milvus_paper/deep1b/base/base.fvecs"
file_size = os.path.getsize(fn)
print("file size: ", file_size)

with open(fn, 'rb') as f:
    dimension, = struct.unpack('i', f.read(4))
    vector_num = file_size // (4 + 4 * dimension)
    print("vector_num: ", vector_num)
    f.seek(0)
    num = min(num, vector_num)
    vectors = numpy.zeros((num, dimension))
    for i in range(num):
        offset = random.randint(0, vector_num)
        f.seek(offset * (4 + 4 * dimension))
        f.read(4)
        vectors[i] = struct.unpack('f' * dimension, f.read(dimension * 4))
    lens = (vectors ** 2).sum(-1)
    for length in lens:
        print("length: ", length)
