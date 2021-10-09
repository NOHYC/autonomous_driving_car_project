#!/usr/bin/env python3
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
cv_image = np.empty(shape=[0])
# usb 카메라로 부터 받은 image 토픽을 처리하는 콜백 함수 정의
def img_callback(data):
    global cv_image
    # 카메라로부터 받은 img 데이터를 OpenCV로 처리를 할 수 있게 변환해주는 함수 사용
    cv_image = bridge.imgmsg_to_cv2(data,'bgr8')
    print(cv_image.shape)
    print(cv2.__version__)
    #cv2.imshow("original", cv_image)
# 노드 생성
rospy.init_node('image_listener', anonymous=True)
# 구독자 생성
rospy.Subscriber("/detections_image_topic", Image, img_callback)
  
while not rospy.is_shutdown():
    if cv_image.size != (360*640*3):
        continue
    cv2.namedWindow('original')
    cv2.imshow("original", cv_image)
    cv2.waitKey(12)
