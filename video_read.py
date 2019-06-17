import argparse
import json
import logging
import threading
import tensorflow as tf
import cv2
import numpy as np
from kafka import KafkaProducer
from fdfs_client.client import *
from Elastic import Elastic
from config.config import VIDEO_NAME, KAFKA_IP, KAFKA_PORT, TOPIC, KEY, PARTITION, KAFKA_ON, CPU_ON, EVERY_CODE_CPU, TIMES, \
    DOCKER_ID, PROCESS_NUM, ENVIRO, SLEEP_TIME, ES_ON, CUDA_VISIBLE_DEVICES, APISOURCE
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
logger = logging.getLogger('tf-pose')
logger.setLevel(logging.DEBUG)
# file_log = logging.FileHandler("TfPoseEstimator.log")
# file_log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# file_log.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(ch)
# logger.addHandler(file_log)
if ES_ON:
    handler = Elastic()
    logger.addHandler(handler)


def save_to_kafka(now, person_num, is_fall, url, producer, picture):
    """
    保存信息到kafka,字段含义如下
    :param now:
    :param person_num:
    :param is_fall:
    :param url:
    :param producer:
    :param picture:
    :return:
    """
    logger.debug('Upload message save to kafka')
    msg = {
        "equipCode": url,  # 摄像头编号
        "staffChangeTime": now,  # 人员识别时间
        "staffNum": person_num,  # 人员检测（数量）
        "picture": picture,  # 图像url
        "videoUrl": url,  # 采集流url
        'apISource': APISOURCE,  # 模块识别码
        "isFall": is_fall,
    }
    print(msg)
    msg = json.dumps(msg).encode('utf-8')
    future = producer.send(TOPIC, key=KEY.encode('utf-8'), value=msg, partition=PARTITION)
    result = future.get(timeout=6)
    logger.debug(result)


def save_img_to_dfs(image):
    """
    保存图片到FastDFS分布式文件系统里
    :param image: 图片矩阵
    :return: 上传文件系统返回参数
    """
    logger.debug('Start to Save the picture')
    client = Fdfs_client(get_tracker_conf("config/fdfs_client.conf"))
    image_encode = cv2.imencode(".jpg", image)[1]
    data_encode = np.array(image_encode)
    str_encode = data_encode.tostring()
    try:
        ret = client.upload_by_buffer(str_encode)
        if 'successed' in ret.get('Status'):
            logger.debug('Save the picture successfully')
            return ret
    except Exception as e:
        logger.debug(e)
        logger.debug('Save the picture failed')


def main(path, producer):
    """
    主函数, 图像识别
    :param path: 视频流地址
    :param producer: kafka是否开启
    :return:
    """
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    tf_config = None
    if CPU_ON and EVERY_CODE_CPU:
        tf_config = tf.ConfigProto(device_count={"CPU": EVERY_CODE_CPU},  # limit to num_cpu_core CPU usage
                                   inter_op_parallelism_threads=1,
                                   intra_op_parallelism_threads=1,
                                   log_device_placement=True)

    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), tf_config=tf_config)
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), tf_config=tf_config)
    logger.debug('cam read+')
    cam = cv2.VideoCapture(path)
    last_person_num = 0
    fps = cam.get(cv2.CAP_PROP_FPS)
    logger.debug('Video url is: {}'.format(cam.isOpened()))
    logger.debug('FPS:{}'.format(fps))
    while cam.isOpened():
        ret_val, image = cam.read()
        image2 = image
        logger.debug('image process+')
        try:
            image = cv2.resize(image, (0, 0), fx=TIMES, fy=TIMES, interpolation=cv2.INTER_CUBIC)
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            logger.debug('postprocess+')
            image, person_num, is_fall = TfPoseEstimator.draw_humans(image2, humans, imgcopy=False)
            if person_num == 0:
                logger.debug('url:{}  Now people is 0,sleep 5 second'.format(path))
                time.sleep(SLEEP_TIME)
                continue
            now = time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))
            if last_person_num != person_num:
                logger.debug('Change ! Now address : {} , Time : {} , Peopel : {}'.format(path, now, person_num))
                if producer:
                    res = save_img_to_dfs(image)
                    picture = str(res.get('Remote file_id'), encoding="utf-8") if res.get('Remote file_id') else ""
                    print(picture)
                    save_to_kafka(now, person_num, is_fall, path, producer, picture)
            logger.debug('Now address : {} , Time : {} , Peopel : {}'.format(path, now, person_num))
            logger.debug('finished+')
            last_person_num = person_num
        except Exception as ex:
            logger.error(ex)
            logger.error('No video')
            logger.error('Now address : {}'.format(path))
            logger.error('No video, This is restarting')
    else:
        logger.error('Video url is bad, url:{}'.format(path))
        logger.error('This is restarting')
        main(path, producer)


def start():
    producer = None
    if KAFKA_ON:
        ip_port = '{}:{}'.format(KAFKA_IP, KAFKA_PORT)
        producer = KafkaProducer(bootstrap_servers=ip_port)
    video_mes = VIDEO_NAME
    if ENVIRO and os.environ[DOCKER_ID]:
        video_mes = video_mes[
                    int(os.environ[DOCKER_ID]) * PROCESS_NUM - PROCESS_NUM:int(os.environ[DOCKER_ID]) * PROCESS_NUM]
    for path in video_mes:
        t = threading.Thread(target=main, args=(path, producer))
        t.start()


if __name__ == '__main__':
    start()
