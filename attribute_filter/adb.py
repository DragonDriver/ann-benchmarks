from __future__ import absolute_import
from base import BaseANN
import psycopg2
from psycopg2 import extras
import asyncio
import asyncpg
import time

class AnalyticDB(BaseANN):
    def __init__(
        self,
        dataset,
        host='mygpdbpub.gpdb.rds.aliyuncs.com',
        useless='useless',
        database='postgres',
        user='annbench',
        password='Fantast1c',
        port=5432,
    ):
        self._database = database
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        while True:
            try:
                self._conn = psycopg2.connect(
                    database=self._database,
                    user=self._user,
                    password=self._password,
                    host=self._host,
                    port=self._port,
                )
                break
            except Exception as e: # retry
                print('retry to connect: {}'.format(host))
                time.sleep(1)
                continue
        self._conn.autocommit = True
        self._cursor = self._conn.cursor()
        self._table_name = dataset.replace('-', '_') + '_test_filter'
        self._already_nums = 0 # batch fit

    def _get_database_size_of_disk(self):
        # pg 可以统计数据库占用空间、表占用空间、索引占用空间
        # 因为我们实验只会在 postgres 这个数据库里进行操作
        # 统计数据库占用空间就行了
        sql = "select pg_database_size('%s')" % (self._database)
        print(sql)
        self._cursor.execute(sql)
        database_size_of_disk = self._cursor.fetchall()
        print(database_size_of_disk[0][0])
        return database_size_of_disk[0][0] / 1024

    def get_memory_usage(self):
        return self._get_database_size_of_disk()

    def _table_exist(self):
        exist_sql = "select count(*) from pg_class where relname = '{}'".format(self._table_name)
        self._cursor.execute(exist_sql)
        exist_rows = self._cursor.fetchall()
        assert len(exist_rows) == 1
        return exist_rows[0][0] != 0

    def _has_index(self):
        has_index_sql = "select relhasindex from pg_class where relname = '{}'".format(self._table_name)
        self._cursor.execute(has_index_sql)
        has_index_rows = self._cursor.fetchall()
        return has_index_rows[0][0]

    def _get_row_nums(self):
        count_sql = "select count(*) from {}".format(self._table_name)
        self._cursor.execute(count_sql)
        count_rows = self._cursor.fetchall()
        return count_rows[0][0]

    def _drop_table(self):
        drop_sql = 'drop table {}'.format(self._table_name)
        print(drop_sql)
        self._cursor.execute(drop_sql)
        self._conn.commit()

    def _create_table(self):
        create_sql = "create table {} (id int, field int, vector real[])".format(self._table_name)
        print(create_sql)
        self._cursor.execute(create_sql)
        self._conn.commit()

    def _create_index(self, dimension):
        index_sql = "create index on {} using ann(vector) with (dim={})".format(self._table_name, dimension)
        print(index_sql)
        t = time.time()
        self._cursor.execute(index_sql)
        self._conn.commit()
        print("create index, time cost: ", time.time() - t)
        index_sql = "create index on field"
        print(index_sql)
        t = time.time()
        self._cursor.execute(index_sql)
        self._conn.commit()
        print("create index, time cost: ", time.time() - t)

    def already_fit(self, total_num):
        if self._table_exist():
            if self._get_row_nums() >= total_num and self._has_index():
                return True
        return False

    def support_batch_fit(self):
        return True

    def _fit_with_offset(self, X, nums, offset):
        row_nums = len(X)
        step = 10000
        insert_sql = "insert into {}(id, field, vector) values %s".format(self._table_name)
        for i in range(0, row_nums, step):
            end = min(i + step, row_nums)
            rows = [(j + offset, nums[j], X[j].tolist()) for j in range(i, end)]
            psycopg2.extras.execute_values(self._cursor, insert_sql, rows)
            self._conn.commit()

    def batch_fit(self, X, nums, total_num):
        if self._already_nums == 0:
            if self._table_exist():
                self._drop_table()
            self._create_table()
        assert self._already_nums < total_num
        self._fit_with_offset(X, nums, self._already_nums)
        self._already_nums += len(X)
        if self._already_nums >= total_num:
            self._create_index(X.shape[1])

    def fit(self, X, nums):
        if self._table_exist():
            self._drop_table()
        self._create_table()
        self._fit_with_offset(X, nums, 0)
        self._create_index(X.shape[1])

    def set_query_arguments(self, useless='useless'):
        pass

    def query(self, v, n, upper_bound):
        query_sql = """select id
                       from {}
                       where field < {}
                       order by vector <-> array{}
                       limit {}
        """.format(self._table_name, upper_bound, v.tolist(), n)
        self._cursor.execute(query_sql)
        ids = self._cursor.fetchall()
        return ids

    def __str__(self):
        return 'AnalyticDB for PostgreSQL, machine: %s' % (self._host)

    def done(self):
        if self._table_exist():
            self._drop_table()
        self._conn.close()

class AnalyticDBAsync(AnalyticDB):
    def __init__(
        self,
        dataset,
        host='mygpdbpub.gpdb.rds.aliyuncs.com',
        useless='useless',
        database='postgres',
        user='annbench',
        password='Fantast1c',
        port=5432,
    ):
        AnalyticDB.__init__(
            self,
            dataset,
            host,
            useless,
            database,
            user,
            password,
            port,
        )
        self._el = asyncio.get_event_loop()
        db_setting = {
            'database': database,
            'user': user,
            'password': password,
            'host': host,
            'port': port,
        }
        self._db_pool = self._el.run_until_complete(asyncpg.create_pool(
            **db_setting
        ))

    async def batch_insert(self, records):
        async with self._db_pool.acquire() as conn:
            await conn.copy_records_to_table(self._table_name, records=records)

    def _fit_with_offset(self, X, nums, offset):
        async def insert_records():
            row_nums = len(X)
            step = 10000
            coros = []
            for i in range(0, row_nums, step):
                end = min(i + step, row_nums)
                rows = [(j + offset, nums[j], X[j].tolist()) for j in range(i, end)]
                coros.append(self.batch_insert(rows))
            await asyncio.gather(*coros)

        self._el.run_until_complete(insert_records())

    def batch_query(self, X, n, upper_bound):
        # query_sql = "select id from {} order by vector <-> array$1 limit {}".format(self._table_name, n)

        rows = list(range(len(X)))
        async def query_async():
            async def single_query(i):
                async with self._db_pool.acquire() as conn:
                    sql = "select id from {} where field < {} order by vector <-> array{} limit {}".format(self._table_name, upper_bound, X[i].tolist(), n)
                    rows[i] = await conn.fetch(sql)
                    # rows[i] = await conn.fetch(query_sql, X[i].tolist())

            coros = [single_query(i) for i in range(len(X))]
            await asyncio.gather(*coros)
            self._res = rows

        self._el.run_until_complete(query_async())

    def get_batch_results(self):
        batch_results = []
        for topk in self._res:
            batch_results.append([t.get('id') for t in topk])
        return batch_results

    def __str__(self):
        return 'Asyncorization AnalyticDB for PostgreSQL, machine: {}'.format(self._host)
