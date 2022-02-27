import os
import cv2
from pathlib import Path

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if __name__ == '__main__':
    bag = rosbag.Bag("/home/chao/Dropbox/20220227_211624.bag", "r")
    bridge = CvBridge()
    count = 0
    topics = [
        "/device_0/sensor_0/Infrared_1/image/data",
        "/device_0/sensor_0/Infrared_2/image/data"
    ]

    data_dir = Path("/home/chao/Documents/realsense")
    left_dir = data_dir / "infra1"
    right_dir = data_dir / "infra2"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    for topic, msg, t in bag.read_messages(topics=topics):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if topic == topics[0]:
            cv2.imwrite(str(left_dir / f"image{count:06d}.png"), cv_img)
            count += 1
            print(count)
        elif topic == topics[1]:
            cv2.imwrite(str(right_dir / f"image{count:06d}.png"), cv_img)
    bag.close()