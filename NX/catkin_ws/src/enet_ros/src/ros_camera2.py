#!/usr/bin/env python
# -*- coding: utf-8 -*-
    
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
# OpenCV를 ROS에서 사용할 수 있게 해주는 모듈
from cv_bridge import CvBridge
print("connect")
bridge = CvBridge()
# 이미지를 담을 빈 배열 생성
cv_detect_image = np.empty(shape=[0])
cv_ddpg_image = np.empty(shape=[0])
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fps = cap.get(cv2.CAP_PROP_FPS)

# usb 카메라로 부터 받은 image 토픽을 처리하는 콜백 함수 정의

def detect_img_callback(data):
    global cv_detect_image
    # 카메라로부터 받은 img 데이터를 OpenCV로 처리를 할 수 있게 변환해주는 함수 사용
    cv_detect_image = bridge.imgmsg_to_cv2(data, "rgb8")

def ddpg_img_callback(data):
    global cv_ddpg_image
    # 카메라로부터 받은 img 데이터를 OpenCV로 처리를 할 수 있게 변환해주는 함수 사용
    cv_ddpg_image = bridge.imgmsg_to_cv2(data, "rgb8")


# 노드 생성
rospy.init_node('image_visualizer')
# 구독자 생성
rospy.Subscriber("/TFF/detections_image_topic", Image, detect_img_callback)
rospy.Subscriber("/TFF/ddpg_image_topic", Image, ddpg_img_callback)
out1 = cv2.VideoWriter('cv_detect_image.avi', fourcc, 30, (int(480), int(360)))
out2 = cv2.VideoWriter('cv_ddpg_image.avi', fourcc, 30, (int(480), int(360)))
  
while not rospy.is_shutdown():
    if cv_detect_image.size != (360*480*3):
        continue

    
    cv2.namedWindow('detect_image')
    cv2.imshow("detect_image", cv_detect_image)
    cv2.imshow("ddpg_image", cv_ddpg_image)
    out1.write(cv_detect_image)
    out2.write(cv_ddpg_image)
    #print("why does it work?")
    cv2.waitKey(25)
