import argparse
import json
import logging
import os
import threading
import time
import tensorflow as tf
import cv2
from kafka import KafkaProducer
from fdfs_client.client import *
from Elastic import Elastic
from config.config import VIDEO_NAME, IP_PORT, TOPIC, KEY, PARTITION, KAFKA_ON, CPU_ON, EVERY_CODE_CPU, TIMES, \
    DOCKER_ID, PROCESS_NUM, ENVIRO, SLEEP_TIME, ES_ON
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    msg = {
        "equipCode": 1,  # 摄像头编号
        "staffChangeTime": now,  # 人员识别时间
        "staffNum": person_num,  # 人员检测（数量）
        "videoUrl": url,  # 采集流url
        'apISource': '1',  # 模块识别码
        "isFall": is_fall,
        "picture": picture,
    }
    msg = json.dumps(msg).encode('utf-8')
    future = producer.send(TOPIC, key=KEY.encode('utf-8'), value=msg, partition=PARTITION)
    result = future.get(timeout=6)
    logger.debug(result)


def save_img_to_DFS(image):
    client = Fdfs_client(get_tracker_conf("./fdfs_client.conf"))
    cv2.imwrite("person.png", image)
    try:
        ret = client.upload_by_filename("person.png")
        if 'successed' in ret.get['Status']:
            logger.debug('Save the picture successfully')
            os.remove('person.png')
            return ret
    except Exception as e:
        logger.debug(e)
        logger.debug('Save the picture failed')


def main(path, producer):
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
    print('Video url is：', cam.isOpened())
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
                logger.debug('Now people is 0,sleep 5 second')
                time.sleep(SLEEP_TIME)
                continue
            now = time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))
            if last_person_num != person_num:
                logger.debug('Change ! Now address : {} , Time : {} , Peopel : {}'.format(path, now, person_num))
                if producer:
                    res = save_img_to_DFS(image)
                    picture = str(res.get('Remote file_id'), encoding="utf-8") if res.get('Remote file_id') else ""
                    save_to_kafka(now, person_num, is_fall, path, producer, picture)
            logger.debug('Now address : {} , Time : {} , Peopel : {}'.format(path, now, person_num))
            logger.debug('finished+')
            last_person_num = person_num
        except Exception as e:
            logger.error(e)
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
        producer = KafkaProducer(bootstrap_servers=IP_PORT)
    video_mes = VIDEO_NAME
    if ENVIRO and os.environ[DOCKER_ID]:
        video_mes = video_mes[
                    int(os.environ[DOCKER_ID]) * PROCESS_NUM - PROCESS_NUM:int(os.environ[DOCKER_ID]) * PROCESS_NUM]
    for path in video_mes:
        t = threading.Thread(target=main, args=(path, producer))
        t.start()


if __name__ == '__main__':
    start()
