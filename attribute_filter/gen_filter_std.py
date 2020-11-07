import h5py
import numpy
import random
import milvus
import os

DISTINCT_NUMS = 100000
SELECTIVITY = 0.95
UPPER_BOUND = round(DISTINCT_NUMS - DISTINCT_NUMS * SELECTIVITY)

DATA_SIZE = 10000000
QUERY_NUM = 10000

group_nums = (DATA_SIZE + DISTINCT_NUMS - 1) // (DISTINCT_NUMS)
print("group_nums: ", group_nums)
int_fn = '../data/attribute_filter/sift-%d-%d-intfield.txt' % (DATA_SIZE, QUERY_NUM)
distinct_values_grps = []
if not os.path.isfile(int_fn):
    for i in range(group_nums):
        distinct_values = random.sample(range(DISTINCT_NUMS), DISTINCT_NUMS)
        distinct_values_grps.append(distinct_values)
    print('generate distinct values done ...')
    with open(int_fn, 'w') as intf:
        for i in range(group_nums):
            for int_field in distinct_values_grps[i]:
                intf.write(str(int_field) + '\n')
    print('write distinct values done ...')
else:
    with open(int_fn, 'r') as f:
        nums = [int(line) for line in f.readlines()]
    print('read distinct values done ...')
    for i in range(group_nums):
        distinct_values = nums[i * DISTINCT_NUMS : (i + 1) * DISTINCT_NUMS]
        # print(len(distinct_values))
        distinct_values_grps.append(distinct_values)

fn = '../data/sift-%d-%d.hdf5' % (DATA_SIZE, QUERY_NUM)
ds = h5py.File(fn, 'r')
dimension = len(ds['train'][0])

client = milvus.Milvus(host='localhost', port='19530', try_connect=False, pre_ping=False)
collection_name = 'gen_filter_std'
nlist = 16
nprobe = nlist
collection_params = {
    'collection_name': collection_name,
    'dimension': dimension,
    'index_file_size': 256,
    'metric_type': milvus.MetricType.L2,
}
print(collection_params)
status, has_table = client.has_collection(collection_name)
if has_table:
    client.drop_collection(collection_name)
client.create_collection(collection_params)

# vectors = numpy.array(ds['train'])
for i in range(group_nums):
    filter_vector_ids = []
    filter_vectors = []
    distinct_values = distinct_values_grps[i]
    # print('len(distinct_values): ', len(distinct_values))
    # print('len(filter_vectors): ', len(filter_vectors))
    for idx, int_value in enumerate(distinct_values):
        global_idx = i * DISTINCT_NUMS + idx
        if int_value < UPPER_BOUND:
            filter_vector_ids.append(global_idx)
            filter_vectors.append(numpy.array(ds['train'][global_idx]).tolist())
    # print('len(filter_vectors): ', len(filter_vectors))
    print('filter {}th gropu vectors done ...'.format(i))
    step = 1000
    records_len = len(filter_vectors)
    print("records_len: ", records_len)
    print("upper_bound: ", UPPER_BOUND)
    for j in range(0, records_len, step):
        end = min(j + step, records_len)
        status, ids = client.insert(
            collection_name=collection_name,
            records=filter_vectors[j:end],
            ids=filter_vector_ids[j:end],
        )
        if not status.OK():
            raise Exception("Insert failed. {}".format(status))
index_type = getattr(milvus.IndexType, 'IVF_FLAT')
index_params = {'nlist': nlist}
top_k = 50
search_params = {'nprobe': nprobe}
status = client.create_index(collection_name, index_type, params=index_params)
if not status.OK():
    raise Exception("create index failed. {}".format(status))
qs = numpy.array(ds['test'])
query_records = []
for q in qs:
    query_records.append(q.tolist())
status, results = client.search(collection_name, top_k, query_records, params=search_params)
if not status.OK():
    raise Exception("Search failed. {}".format(status))
std_fn = '../data/attribute_filter/sift-%d-%d-%f-std.txt' % (DATA_SIZE, QUERY_NUM, SELECTIVITY)
with open(std_fn, 'w') as f:
    print([result.distance for result in results[0]])
    print([result.distance for result in results[-1]])
    for r in results:
        neighbors = [result.id for result in r]
        content = str(neighbors)[1:-1]
        f.write(content + '\n')

ds.close()
client.close()
