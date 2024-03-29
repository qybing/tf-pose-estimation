VIDEO_NAME = ["rtmp://10.10.6.213:11936/live/1", "rtmp://10.10.6.213:11936/live/2", "rtmp://10.10.6.213:11936/live/3"]
# 服务版本号
VERSION = 1
APISOURCE = 'PERSON_SAFE'

# kafka配置
KAFKA_ON = False
KAFKA_IP = "10.10.6.213"
KAFKA_PORT = "9092"
# kafka 消息配置
TOPIC = "TOPIC_IMAGE_PERSON_RECON"
KEY = 'Pose'
PARTITION = 0

# CPU 配置
CPU_ON = False
EVERY_CODE_CPU = 2

# 设置调用那些GPU
CUDA_VISIBLE_DEVICES = "0"

# 图像缩放
TIMES = 0.7

# 每台电脑docker 标识
ENVIRO = False
DOCKER_ID = 'TASK_SLOT'
# 每台docker 处理的视频流几条
PROCESS_NUM = 3

# 休息时间 默认5秒
SLEEP_TIME = 5

# Elastic 设置 账号密码
ES_ON = False
ES_HOST = '10.10.6.213'
ES_PORT = 9200
IS_USER = False
USER_NAME = None
PASSWORD = None
# 索引，类型
INDEX = 'tf-pose'
DOC_TYPE = 'POSE'
