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