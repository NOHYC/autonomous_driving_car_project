#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import glob 
import pickle 
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
# OpenCV를 ROS에서 사용할 수 있게 해주는 모듈
from cv_bridge import CvBridge
bridge = CvBridge()
# 이미지를 담을 빈 배열 생성
#cv_image = np.empty(shape=[0])

rospy.init_node('image_pulisher', anonymous=True)
camera_pub = rospy.Publisher('/TFF/camera_topic', Image, queue_size=10)
def get_cameramat_dist(filename): 
    f = open(filename, 'rb') 
    mat, dist, rvecs, tvecs = pickle.load(f) 
    f.close() 
    #print("camera matrix") 
    #print(mat) 
    #print("distortion coeff") 
    #print(dist) 
    return mat,dist 


def main(): 
    mat, dist = get_cameramat_dist("/home/nohs/catkin_ws/src/enet_ros/src/cam_calib.pkl") 
    cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=480,format=BGR ! appsink")
    ret, frame = cap.read()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("width:",width,"height:",height)
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #fps = cap.get(cv2.CAP_PROP_FPS)
    frame = cv2.flip(frame, -1) 
    rsz = cv2.resize(frame, dsize=(640,480)) 
    gray = cv2.cvtColor(rsz, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2] 
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mat,dist,(w,h),0,(w,h))
    #out = cv2.VideoWriter('out.avi', fourcc, fps, (int(480), int(360))) 
    while(True): 
        ret, frame = cap.read() 
        frame = cv2.flip(frame,-1)
        #print(frame.shape) 
        rsz = cv2.resize(frame, dsize=(640,480))
        gray = rsz
# undistort 
        mapx,mapy = cv2.initUndistortRectifyMap(mat,dist,None,newcameramtx,(w,h),5) 
        res = cv2.remap(gray,mapx,mapy,cv2.INTER_LINEAR) 
# crop the image 
        x,y,w,h = roi 
        res = res[y:y+h, x:x+w]
        res = cv2.resize(res,(480,360))
        #out.write(res)
        #cv2.imshow('res',res)
        cv_image = bridge.cv2_to_imgmsg(res,'bgr8') 
        camera_pub.publish(cv_image)
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break
    cap.release() 
    cv2.destroyAllWindows()
if __name__ == "__main__": 
    main()
