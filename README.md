## 人体姿态识别
### windows
```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd tf-openpose
$ python3 setup.py install
```
配置视频流源
```bash
$ cd config
$ python3 video_read.py
```
Docker 环境部署
```bash
$ cd tf-pose-estimation
$ docker build -t tf-pose:v1 .
$ docker run tf-pose
```
Docker swarm 集群部署
创建 docker config 文件
```bash
$ docker config create pose config.py
$ docker service create --name test --config source=mytest,target=/opt/code/tf-pose-estimation/config/config.py --replicas 3 qiaoyanbing/tf-pose:v5
```