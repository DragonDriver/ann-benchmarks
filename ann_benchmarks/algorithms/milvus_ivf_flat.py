from __future__ import absolute_import
import json
import time
import milvus
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class MilvusIVFFLAT(BaseANN):
    def __init__(self, metric, dataset, index_type, nlist):
        self._index_param = {'nlist': nlist}
        self._search_param = {'nprobe': None}
        self._metric = {'angular': milvus.MetricType.IP, 'euclidean': milvus.MetricType.L2}[metric]
        self._milvus = milvus.Milvus(host='localhost', port='19530', try_connect=False, pre_ping=False)
        # import uuid
        # self._table_name = 'test_' + str(uuid.uuid1()).replace('-', '_')
        self._table_name = dataset.replace('-', '_')
        self._index_file_size = 2048
        postfix = '_' + str(metric) + '_' + str(index_type) + '_' + str(nlist) + '_' + str(self._index_file_size)
        self._table_name += postfix
        self._table_name.replace('-', '_')
        self._index_type = index_type

        # batch fit
        self._already_nums = 0

        # batch search
        self._res = None

    def get_memory_usage(self):
        _, exist = self._milvus.has_collection(self._table_name)
        if not exist:
            return 0.0

        _, stats = self._milvus.get_collection_stats(self._table_name)
        size = 0
        for p in stats["partitions"]:
            for s in p["segments"]:
                size += int(s["data_size"])

        return size / 1024

    def support_batch_fit(self):
        return True

    def already_fit(self, total_num):
        print("table name: ", self._table_name)
        status, has_table = self._milvus.has_collection(self._table_name)
        if has_table:
            status, table_stats = self._milvus.get_collection_stats(self._table_name)
            if table_stats:
                row_nums = table_stats['row_count']
                if row_nums >= total_num:
                    # self._already_nums == row_nums
                    status, index_info = self._milvus.get_index_info(self._table_name)
                    return index_info and len(index_info.params) != 0
        print('not already fit yet...')
        return False

    def batch_fit(self, X, total_num):
        assert self._already_nums < total_num

        if self._metric == milvus.MetricType.IP:
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if self._already_nums == 0:
            status, has_table = self._milvus.has_collection(self._table_name)
            if has_table:
                print("drop table...")
                self._milvus.drop_collection(self._table_name)
            print("create table...")
            self._milvus.create_collection({
                'collection_name': self._table_name, 'dimension': X.shape[1],
                'index_file_size': self._index_file_size, 'metric_type': self._metric}
            )

        vector_ids = [id for id in range(self._already_nums, self._already_nums + len(X))]
        records = X.tolist()
        records_len = len(records)
        step = 20000
        for i in range(0, records_len, step):
            end = min(i + step, records_len)
            status, ids = self._milvus.insert(collection_name=self._table_name, records=records[i:end], ids=vector_ids[i:end])
            if not status.OK():
                raise Exception("Insert failed. {}".format(status))
        self._milvus.flush([self._table_name])
        self._already_nums += records_len

        if self._already_nums == total_num:
            index_type = getattr(milvus.IndexType, self._index_type)  # a bit hacky but works
            status = self._milvus.create_index(self._table_name, index_type, params=self._index_param)
            if not status.OK():
                raise Exception("Create index failed. {}".format(status))

    def fit(self, X):
        # if self._metric == 'angular':
        if self._metric == milvus.MetricType.IP:
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        status, has_table = self._milvus.has_collection(self._table_name)
        if has_table:
            self._milvus.drop_collection(self._table_name)
        self._milvus.create_collection({
            'collection_name': self._table_name, 'dimension': X.shape[1],
            'index_file_size': self._index_file_size, 'metric_type': self._metric}
        )

        vector_ids = [id for id in range(len(X))]
        records = X.tolist()
        records_len = len(records)
        step = 20000
        for i in range(0, records_len, step):
            end = min(i + step, records_len)
            status, ids = self._milvus.insert(collection_name=self._table_name, records=records[i:end], ids=vector_ids[i:end])
            if not status.OK():
                raise Exception("Insert failed. {}".format(status))
        self._milvus.flush([self._table_name])

        index_type = getattr(milvus.IndexType, self._index_type)  # a bit hacky but works
        status = self._milvus.create_index(self._table_name, index_type, params=self._index_param)
        if not status.OK():
            raise Exception("Create index failed. {}".format(status))
#         self._milvus_id_to_index = {}
#         self._milvus_id_to_index[-1] = -1 #  -1 means no results found
#         for i, id in enumerate(ids):
#             self._milvus_id_to_index[id] = i

    def set_query_arguments(self, nprobe):
        if nprobe > self._index_param['nlist']:
            print('warning! nprobe > nlist')
            nprobe = self._index_param['nlist']
        self._search_param['nprobe'] = nprobe

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        v = v.tolist()
        future = self._milvus.search(collection_name=self._table_name, query_records=[v], top_k=n, params=self._search_param, _async=True)
        return future

    def handle_query_list_result(self, query_list):
        handled_result = []
        t0 = time.time()
        for index, query in enumerate(query_list):
            total, v, future = query
            status, results = future.result()
            if not status.OK():
                raise Exception("[Search] search failed: {}".format(status.message))

            if not results:
                raise Exception("Query result is empty")
            # r = [self._milvus_id_to_index[z.id] for z in results[0]]
            results_ids = []
            for result in results[0]:
                results_ids.append(result.id)
            handled_result.append((total, v, results_ids))
            # return results_ids
        return time.time() - t0, handled_result

    def batch_query(self, X, n):
        status, results = self._milvus.search(collection_name=self._table_name, query_records=X, top_k=n, params=self._search_param)
        if not status.OK():
            raise Exception("[Search] search failed: {}".format(status.message))

        self._res = results

    def get_batch_results(self):
        batch_results = []
        for r in self._res:
            batch_results.append([result.id for result in r])

        return batch_results

    def __str__(self):
        return 'MilvusIVFFLAT(index={}, index_param={}, search_param={})'.format(self._index_type, self._index_param, self._search_param)

    def done(self):
        print("Delete table ", self._table_name, "......")
        self._milvus.drop_collection(self._table_name)


class MilvusIVFSQ8(MilvusIVFFLAT):
    def __str__(self):
        return 'MilvusIVFSQ8(index={}, index_param={}, search_param={})'.format(self._index_type, self._index_param, self._search_param)


class MilvusIVFSQ8H(MilvusIVFFLAT):
    def __str__(self):
        return 'MilvusIVFSQ8H(index={}, index_param={}, search_param={})'.format(self._index_type, self._index_param, self._search_param)
