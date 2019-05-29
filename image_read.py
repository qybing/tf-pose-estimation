import argparse
import time

from config.config import IMAGE_NAME
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common
import cv2

parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
parser.add_argument('--camera', type=int, default=0)
# parser.add_argument('--image', type=str, default='./images/timg.jpg')
parser.add_argument('--image', type=str, default=IMAGE_NAME)
parser.add_argument('--resize', type=str, default='432x368',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')

parser.add_argument('--model', type=str, default='mobilenet_thin',
                    help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')
args = parser.parse_args()
w, h = model_wh(args.resize)
image = common.read_imgfile(args.image, None, None)

e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

# image = cv2.resize(image, (0, 0), fx=2, fy=2)
image, person_num = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
now = time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))
print('Now Time : {}, People : {}'.format(now, person_num))
# cv2.imshow('result', image)
# cv2.waitKey(0)


# print(image)
