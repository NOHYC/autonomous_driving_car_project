#!/usr/bin/env python3

from __future__ import division
#### ros import 
import glob 
import pickle
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
#from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray #c
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
import cv2
from cv_bridge import CvBridge, CvBridgeError

# python import 
import os
import argparse
import torch
from model_ENET_SAD import ENet_SAD
from utils.prob2lines import getLane
from utils.transforms import *
import time
from multiprocessing import Process, JoinableQueue, SimpleQueue
from threading import Lock

import math

package = RosPack()
package_path = package.get_path('enet_ros') #off

#img_size = (640, 368)
img_size = (480, 360) ###

net = ENet_SAD(img_size, sad=False) #off
# CULane mean, std 
mean=(0.3598, 0.3653, 0.3662)
std=(0.2573, 0.2663, 0.2756)
transform_img = Resize(img_size)
transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))
weights_path = os.path.join(package_path, 'weight', 'exp_hancom_best1.pth')

save_dict = torch.load(weights_path, map_location='cpu')
net.load_state_dict(save_dict['net']) #off
net.eval() #off
net.cuda() #off

rospy.loginfo("Found weights, loading %s", weights_path)
if not os.path.isfile(weights_path): #off
    raise IOError(('{:s} not found.').format(weights_path))
class DetectorManager():

    def __init__(self):
        self.published_image_topic = rospy.get_param('~detections_image_topic')
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.publish_image = rospy.get_param('~publish_image')

        # Load CvBridge
        self.bridge = CvBridge()
        self.cv_image = np.empty(shape=[0])
        # Define subscribers
        #self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCb, queue_size = 1, buff_size = 2**24)
        self.pub_viz_1 = rospy.Publisher(self.published_image_topic, Image, queue_size=10)
        self.pub_viz_2 = rospy.Publisher("/TFF/ddpg_image_topic", Image, queue_size=10)
        self.pub_input_ = rospy.Publisher('/TFF/ddpg_camera_input', Float32MultiArray, queue_size=10) #c
        self.input_msg1 = Float32MultiArray()
        self.input_msg2 = Float32MultiArray()
        self.input_msg3 = Float32MultiArray()
        self.input_msg4 = Float32MultiArray()
        self.input_msg5 = Float32MultiArray()
        rospy.loginfo("gpu: "+ str(torch.cuda.is_available()))
        rospy.loginfo("Launched node for object detection")

        # Spin
        #rospy.spin()

    def network(self,enet, img): #off
        seg_pred, exist_pred = enet(img.cuda())[:2] #off
        seg_pred = seg_pred.detach().cpu() #off
        exist_pred = exist_pred.detach().cpu() #off
        return seg_pred, exist_pred #off

    def get_cameramat_dist(self,filename): 
        f = open(filename, 'rb') 
        mat, dist, rvecs, tvecs = pickle.load(f) 
        f.close() 
        return mat,dist

    def visualize1(self,img, seg_pred, exist_pred):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        lane_img = np.zeros_like(img)
        color = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')

        coord_mask = np.argmax(seg_pred, axis=0)

        #for i in range(0, 4):
        for i in range(0, 1):
            if exist_pred[0, i] > 0.5:
                lane_img[coord_mask == (i + 1)] = color[i]

        img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)

        return img, lane_img
     

    def visualize2(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, bin_img = cv2.threshold(img_gray, 75, 255, cv2.THRESH_BINARY) # have to figure out
        kernel = np.ones((3, 3), np.uint8)
        dilation_img = cv2.dilate(bin_img, kernel, iterations=3) # changed
        erosion_img = cv2.erode(dilation_img, kernel, iterations=3)
        


        line_img = np.zeros_like(img_gray)

        input1 = self.custom_draw_line(erosion_img, line_img, -1, -2, 1)
        input2 = self.custom_draw_line(erosion_img, line_img, -3, -2, 2)
        input3 = self.custom_draw_line(erosion_img, line_img, -3, 0, 3)
        input4 = self.custom_draw_line(erosion_img, line_img, -3, 2, 4)
        input5 = self.custom_draw_line(erosion_img, line_img, -1, 2, 5)

        x_goal_draw, y_goal_draw = (input1[0]+input2[0]+input4[0]+input5[0])//4, (input1[1]+input2[1]+input4[1]+input5[1])//4
        y_goal, x_goal = (input1[0]+input2[0]+input4[0]+input5[0]-1440)//4, (input1[1]+input2[1]+input4[1]+input5[1]-960)//4
        theta = math.atan2(-y_goal, x_goal)
        goal_yaw = -(theta*57.3-90) # theta

        self.input_msg1.data = input1 + input2 + input3 + input4 + input5 #c

        img = cv2.addWeighted(src1=line_img, alpha=1., src2=erosion_img, beta=0.3, gamma=0.)
        img = cv2.arrowedLine(img, (line_img.shape[1]//2, line_img.shape[0]), (input3[1], input3[0]), 245, 3)
        img = cv2.arrowedLine(img, (line_img.shape[1]//2, line_img.shape[0]), (y_goal_draw, x_goal_draw), 245, 3)
        return img #s

    def custom_draw_line(self, erosion_img, line_img, m, n, idx):
        a, b = line_img.shape[0], line_img.shape[1]//2
        for i in range(0,120):
            if erosion_img[(line_img.shape[0]-1) +m*i][line_img.shape[1]//2 +n*i] == 255:
                a, b = ((line_img.shape[0]-1) +m*i, line_img.shape[1]//2 +n*i)
            else:
                line_img = cv2.line(line_img, (line_img.shape[1]//2, line_img.shape[0]), (b, a), 76, 3)
                break
        start_point= np.array((line_img.shape[1]//2, line_img.shape[0]))
        detect_point = np.array((b, a))
        dist = np.linalg.norm(start_point - detect_point)
        return [a, b, dist] # a = y, b = x

    def imageCb(self):
        mat, dist = self.get_cameramat_dist("/home/nohs/catkin_ws/src/enet_ros/src/cam_calib.pkl") 
        cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=480,format=BGR ! appsink")
        ret, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        rsz = cv2.resize(frame, dsize=(640,640)) 
        gray = cv2.cvtColor(rsz, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2] 
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mat,dist,(w,h),0,(w,h))
        #frame = self.bridge.imgmsg_to_cv2(frame, "bgr8")

        while(True): 
            ret, frame = cap.read() 

            gray = cv2.resize(frame, dsize=(640,640))
            #gray = rsz
    # undistort 
            mapx,mapy = cv2.initUndistortRectifyMap(mat,dist,None,newcameramtx,(w,h),5) 
            res = cv2.remap(gray,mapx,mapy,cv2.INTER_LINEAR) 
    # crop the image 
            x,y,w,h = roi 
            res = res[y:y+h, x:x+w]
            frame = cv2.resize(res,(480,360))

            loop_start = time.time()
                    
            frame = transform_img({'img': frame})['img']
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = transform_to_net({'img': img})['img'] #off
            x.unsqueeze_(0) #off

            gpu_start = time.time() #off
            seg_pred, exist_pred = self.network(net, x) #off
            gpu_end = time.time() #off

            seg_pred = seg_pred.numpy()[0] #off
            exist_pred = exist_pred.numpy() #off

            exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(1)] #off

            loop_end = time.time()

            img1 = self.visualize1(img, seg_pred, exist_pred)[0] #off
            img2 = self.visualize2(self.visualize1(img, seg_pred, exist_pred)[1])

            image_msg1 = self.bridge.cv2_to_imgmsg(img1, "rgb8")
            image_msg2 = self.bridge.cv2_to_imgmsg(img2, "mono8")

            self.pub_viz_1.publish(image_msg1)
            self.pub_viz_2.publish(image_msg2)

            self.pub_input_.publish(self.input_msg1) #c


            print("gpu_runtime:", gpu_end - gpu_start, "FPS:", int(1 / (gpu_end - gpu_start))) #off
            print("total_runtime:", loop_end - loop_start, "FPS:", int(1 / (loop_end - loop_start))) #off
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cap.release() 
        cv2.destroyAllWindows()



if __name__ == "__main__":
    # Initialize node
    rospy.init_node("detector_manager_node")
    dm = DetectorManager()
    dm.imageCb()
    print("end")
