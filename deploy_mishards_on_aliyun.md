## 安装 docker：

```shell
apt-get remove docker docker-engine docker.io containerd runc
apt-get update
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
apt-key fingerprint 0EBFCD88
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io
```

## MySQL 元数据服务

```shell
docker pull mysql:5.7
docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
mysql -h localhost -uroot -p
# create database milvus;
```

## 挂载 NAS

详见 https://nasnext.console.aliyun.com/cn-hangzhou/filesystem/0ba934af88/mount

```shell
apt-get update && apt-get install -y nfs-common
echo "options sunrpc tcp_slot_table_entries=128" >>  /etc/modprobe.d/sunrpc.conf 
echo "options sunrpc tcp_max_slot_table_entries=128" >>  /etc/modprobe.d/sunrpc.conf
mkdir -p /mnt/nas
mount -t nfs -o vers=4,minorversion=0,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 0ba934af88-tkj3.cn-hangzhou.nas.aliyuncs.com:/ /mnt/nas
df -h | grep aliyun
```

## 节点

```shell
docker pull milvusdb/milvus:0.10.2-cpu-d081520-8a2393
mkdir -p /home/$USER/milvus/conf
cd /home/$USER/milvus/conf
wget https://raw.githubusercontent.com/milvus-io/milvus/0.10.2/core/conf/demo/server_config.yaml
# 修改 server_config.yaml
# 将 meta_uri 改为 mysql://root:123456@<MySQL_server_host IP>:3306/milvus
# 读节点，role -> ro
# 写节点，role -> rw
# cache size 450
# wal false
docker run -d --name milvus_cpu_0.10.2 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:0.10.2-cpu-d081520-8a2393

docker run -d --name milvus_cpu_0.10.2 -p 19530:19530 -p 19121:19121 -v /mnt/nas/index/db:/var/lib/milvus/db -v /home/$USER/milvus/conf:/var/lib/milvus/conf -v /home/$USER/milvus/logs:/var/lib/milvus/logs -v /home/$USER/milvus/wal:/var/lib/milvus/wal milvusdb/milvus:0.10.2-cpu-d081520-8a2393

docker run -d --name milvus_cpu_0.10.2 -p 19530:19530 -p 19121:19121 -v /mnt/nas/for-create/db:/var/lib/milvus/db -v /home/$USER/milvus/conf:/var/lib/milvus/conf -v /home/$USER/milvus/logs:/var/lib/milvus/logs -v /home/$USER/milvus/wal:/var/lib/milvus/wal milvusdb/milvus:0.10.2-cpu-d081520-8a2393
```

## mishards

```shell
apt-get update
apt-get install git
git clone https://github.com/milvus-io/milvus
git checkout 0.10.2
cd milvus/shards
pip install -r requirements.txt
cp mishards/.env.example mishards/.env
# 修改 mishards/.env
nohup python mishards/main.py > run.log 2>&1 & # 后台启动
```

.env:

```
DEBUG=True

WOSERVER=tcp://127.0.0.1:19530
SERVER_PORT=19535
SERVER_TEST_PORT=19888
MAX_WORKERS=50

SQLALCHEMY_DATABASE_URI=mysql+pymysql://root:123456@127.0.0.1:3306/milvus?charset=utf8mb4
SQL_ECHO=False

SQLALCHEMY_DATABASE_TEST_URI=mysql+pymysql://root:123456@127.0.0.1:3306/milvus?charset=utf8mb4
SQL_TEST_ECHO=False

DISCOVERY_STATIC_HOSTS=127.0.0.1
DISCOVERY_STATIC_PORT=19530
```

