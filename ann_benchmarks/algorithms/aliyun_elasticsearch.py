from __future__ import absolute_import
import asyncio
import uuid
import sys
import time
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from ann_benchmarks.algorithms.base import BaseANN


class AliESHNSW(BaseANN):
    def __init__(self, metric, dataset, method_param):
        self._metric = metric
        self._index_name = dataset.replace('-', '_') + str(uuid.uuid1()).replace('-', '_')
        print("index name: ", self._index_name, flush=True)
        self._method_param = method_param
        self._ef = None
        self._field = "vec"
        self._es = AsyncElasticsearch("http://******.public.elasticsearch.aliyuncs.com:9200",
                                      http_auth=("********", "********"), max_retries=5, retry_on_timeout=True, timeout=600)
        self._loop = asyncio.get_event_loop()
        self._fit_count = 0

    def get_memory_usage(self):
        exists = self._loop.run_until_complete(self._es.indices.exists(index=self._index_name))
        if not exists:
            return 0

        stats = self._loop.run_until_complete(self._es.indices.stats(index=self._index_name))
        return int(stats["_all"]["primaries"]["store"]["size_in_bytes"]) / 1024

    def wait_task_empty(self, waiter):
        while True:
            health = self._loop.run_until_complete(self._es.cluster.health())
            pending_tasks = int(health["number_of_pending_tasks"])
            if pending_tasks > 0:
                print("Pending task is {}, <{}> waiting for 10s ....".format(pending_tasks, waiter), flush=True)
                time.sleep(2)
                continue

            break

    def on_start(self, *args, **kwargs):
        dim = kwargs.get("dim", None)
        if dim is None:
            raise ValueError("Param dim is needed")

        _index_body = {
            "mappings": {
                "properties": {
                    self._field: {
                        "type": "proxima_vector",
                        "dim": dim
                    }
                }
            },
            "settings": {
                "index.codec": "proxima",
                "index.vector.algorithm": "hnsw",
                "index.vector.hnsw.builder.neighbor_cnt": self._method_param["M"] * 2,
                "index.vector.hnsw.builder.upper_neighbor_cnt": self._method_param["M"],
                "index.vector.hnsw.builder.efconstruction": self._method_param["efConstruction"],
                "index.refresh_interval": "1s",
                "index.number_of_replicas": 1,
                "index.number_of_shards": 1,
                # "index.write.wait_for_active_shards": "2"
            }
        }

        exist = self._loop.run_until_complete(self._es.indices.exists(index=self._index_name))
        if exist:
            print("Found index {}. delete it".format(self._index_name), flush=True)
            self._loop.run_until_complete(self._es.indices.delete(index=self._index_name))
            time.sleep(10)
        created = self._loop.run_until_complete(self._es.indices.create(index=self._index_name, body=_index_body))

    def support_batch_fit(self):
        return True

    def already_fit(self, total_num):
        exists = self._loop.run_until_complete(self._es.indices.exists(index=self._index_name))
        if not exists:
            return False

        stats = self._loop.run_until_complete(self._es.indices.stats(index=self._index_name))
        count = stats["_all"]["primaries"]["docs"]["count"]
        return count == total_num

    def fit(self, X):
        print("Fit data... already data is {}".format(self._fit_count), flush=True)
        count = X.shape[0]

        for offset in range(0, count, 1000):
            batch_end = min(offset + 1000, count)
            Xp = X[offset: batch_end]
            batch_count = batch_end - offset

            async def gen_action():
                for i, vec in enumerate(Xp.tolist()):
                    yield {
                        "_index": self._index_name,
                        "_id": i + self._fit_count,
                        "_source": {
                            self._field: vec,
                        }
                    }

            success, failed = self._loop.run_until_complete(
                async_bulk(self._es, gen_action(), stats_only=True, index=self._index_name, raise_on_error=True))
            if success < batch_count or failed > 0:
                raise Exception("Create index failed. Total {} vectors, {} success, {} fail".format(batch_count, success, failed))
            self.wait_task_empty("fit.{}".format(self._fit_count))
            self._fit_count += batch_count

    def batch_fit(self, X, total_num):
        if self._fit_count == 0:
            print("Start create index ...")
            dim = X.shape[1]
            self.on_start(dim=dim)

        self.fit(X)
        if self._fit_count == total_num:
            print("| ES | refresh", flush=True)
            self._loop.run_until_complete(self._es.indices.refresh(index=self._index_name, params={"request_timeout": 1800}))
            self.wait_task_empty("fit.refresh")
            print("| ES | flush", flush=True)
            self._loop.run_until_complete(self._es.indices.flush(index=self._index_name, params={"request_timeout": 1800}))
            self.wait_task_empty("fit.flush")
            print("| ES | forcemerge", flush=True)
            self._loop.run_until_complete(self._es.indices.forcemerge(index=self._index_name, params={"request_timeout": 1800}))
            self.wait_task_empty("fit.forcemerge")

            stats = self._loop.run_until_complete(self._es.indices.stats(index=self._index_name))
            count = stats["_all"]["primaries"]["docs"]["count"]
            if count != total_num:
                deleted = stats["_all"]["primaries"]["docs"]["deleted"]
                print("Error: fit data failed. count is {}, but total is {}. {} is deleted ".format(count, total_num,
                                                                                                    deleted),
                      file=sys.stderr, flush=True)
                raise RuntimeError("Fit data failed. stored data count is not equal to total")

    def set_query_arguments(self, ef):
        self._ef = ef

    def query(self, v, n):
        query_params = {
            "query": {
                "hnsw": {
                    self._field: {
                        "vector": v,
                        "size": n,
                        "ef": self._ef
                    }
                }
            },
            "_source": False,
            "from": 0,
            "size": n
        }

        return self._es.search(index=self._index_name, body=query_params)

    def handle_query_list_result(self, query_list):
        t0 = time.time()
        ql = (future for _, _,  future in query_list)
        results = self._loop.run_until_complete(asyncio.gather(*ql))
        handled_result = []

        for q, result in zip(query_list, results):
            total, v, _ = q
            result_ids = [int(doc['_id']) for doc in result['hits']['hits']]
            handled_result.append([total, v, result_ids])

        return time.time() - t0, handled_result

    def done(self):
        exists = self._loop.run_until_complete(self._es.indices.exists(index=self._index_name))
        if exists:
            time.sleep(10)
            print("Deleting index {} ...".format(self._index_name), flush=True)
            self._loop.run_until_complete(self._es.indices.delete(index=self._index_name))
            self.wait_task_empty("done.done")

        self._loop.run_until_complete(self._es.close())
        self._fit_count = 0

    def __str__(self):
        return "Elasticsearch({}, param: {}, search param: {})".format("euclidean", self._method_param, self._ef)
