VIDEO_NAME = [""]

# kafka配置
KAFKA_ON = False
IP_PORT = "10.10.1.9:9092"
# kafka 消息配置
TOPIC = "TOPIC_IMAGE_PERSON_RECON"
KEY = 'Pose'
PARTITION = 0

# CPU 配置
CPU_ON = False
EVERY_CODE_CPU = 2

# 图像缩放
TIMES = 0.7

# 每台电脑docker 标识
ENVIRO = False
DOCKER_ID = None
# 每台docker 处理的视频流几条
PROCESS_NUM = 3

# 休息时间 默认5秒
SLEEP_TIME = 5