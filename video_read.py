import argparse
import json
import logging
import os
import threading
import time
import tensorflow as tf
import cv2
from kafka import KafkaProducer

from config.config import VIDEO_NAME, IP_PORT, TOPIC
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
file_log = logging.FileHandler("TfPoseEstimator.log")
file_log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
file_log.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(file_log)
producer = KafkaProducer(bootstrap_servers=IP_PORT)
fps_time = 100


def save_to_kafka(now, person_num, is_fall, url):
    msg = {
        "equipCode": 1,    # 摄像头编号
        "staffChangeTime": now,   # 人员识别时间
        "staffNum": person_num,  # 人员检测（数量）
        "videoUrl": url,        # 采集流url
        'apISource': '1',       # 模块识别码
        "isFall": is_fall
    }
    msg = json.dumps(msg).encode('utf-8')
    future = producer.send(TOPIC, key=b'Pose', value=msg, partition=0)
    result = future.get(timeout=6)
    logger.debug(result)
    # producer.close()


def main(name, path):
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

    config = tf.ConfigProto(device_count={"CPU": 2},  # limit to num_cpu_core CPU usage
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1,
                            log_device_placement=True)

    if w > 0 and h > 0:
        # e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), tf_config=config)
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(path)
    last_person_num = 0
    fps = cam.get(cv2.CAP_PROP_FPS)
    i = 1
    print('what is：', cam.isOpened())
    logger.debug('FPS:{}'.format(fps))
    while cam.isOpened():
        ret_val, image = cam.read()
        # logger.debug('This is  {}'.format(i))
        # # if i % fps == 0:
        # logger.debug('Processing is {}'.format(i))
        image2 = image
        logger.debug('image process+')
        try:
            image = cv2.resize(image, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            logger.debug('postprocess+')
            image, person_num, is_fall = TfPoseEstimator.draw_humans(image2, humans, imgcopy=False, video_name='video1')
            if person_num == 0:
                logger.debug('Now people is 0,sleep 5 second')
                time.sleep(5)
                continue
            now = time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))
            if last_person_num != person_num:
                logger.debug('Change ! Now address : {} , Time : {} , Peopel : {}'.format(name, now, person_num))
                save_to_kafka(now, person_num, is_fall, name)
            logger.debug('Now address : {} , Time : {} , Peopel : {}'.format(name, now, person_num))
            logger.debug('finished+')
            last_person_num = person_num
        except Exception as e:
            logger.error(e)
            logger.error('No video')
            logger.error('Now address : {}'.format(name))
            logger.error('No video, This is restarting')
        # i = i + 1
    else:
        logger.error('This is restarting')
        main(name, path)


def start():
    video_mes = VIDEO_NAME
    for name, path in video_mes.items():
        t = threading.Thread(target=main, args=(name, path))
        # t = Process(target=main, args=(name, path))
        t.start()


if __name__ == '__main__':
    start()
