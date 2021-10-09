#!/usr/bin/env python3
#### ros import 
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
from std_msgs.msg import Float32MultiArray #c
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
import cv2
from cv_bridge import CvBridge, CvBridgeError

# python import 
import os
import argparse
import time
import math

package = RosPack()
img_size = (480, 360)




class DetectorManager():
    def __init__(self):
        # Load image parameter and confidence threshold
        self.image_topic = rospy.get_param('~image_topic', '/carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation')
        # Load publisher topics
        self.published_image_topic = rospy.get_param('~detections_image_topic')

        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.publish_image = rospy.get_param('~publish_image')

        # Load CvBridge
        self.bridge = CvBridge()
        
        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCb, queue_size = 1, buff_size = 2**24)
        self.pub_viz_ = rospy.Publisher(self.published_image_topic, Image, queue_size=10)
        self.pub_input_ = rospy.Publisher('/TFF/ddpg_camera_input', Float32MultiArray, queue_size=10) #c
        self.input_msg1 = Float32MultiArray()
        self.input_msg2 = Float32MultiArray()
        self.input_msg3 = Float32MultiArray()
        self.input_msg4 = Float32MultiArray()
        self.input_msg5 = Float32MultiArray()
        
        rospy.loginfo("Launched node for object detection")
        rospy.spin()


    def visualize(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, bin_img = cv2.threshold(img_gray, 91, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion_img = cv2.erode(bin_img, kernel, iterations=3)
        dilation_img = cv2.dilate(erosion_img, kernel, iterations=3)


        line_img = np.zeros_like(img_gray)

        input1 = self.custom_draw_line(dilation_img, line_img, -1, -2, 1)
        input2 = self.custom_draw_line(dilation_img, line_img, -3, -2, 2)
        input3 = self.custom_draw_line(dilation_img, line_img, -3, 0, 3)
        input4 = self.custom_draw_line(dilation_img, line_img, -3, 2, 4)
        input5 = self.custom_draw_line(dilation_img, line_img, -1, 2, 5)
        x_average = (input1[0] + input2[0] + input3[0] + input4[0] + input5[0]) // 5
        x_goal_draw, y_goal_draw = (input1[0]+input2[0]+input4[0]+input5[0])//4, (input1[1]+input2[1]+input4[1]+input5[1])//4
        y_goal, x_goal = (input1[0]+input2[0]+input4[0]+input5[0]-1440)//4, (input1[1]+input2[1]+input4[1]+input5[1]-960)//4
        theta = math.atan2(-y_goal, x_goal)
        goal_yaw = -(theta*57.3-90) 


        self.input_msg1.data = input1 + input2 + input3 + input4 + input5 
        
        img = cv2.addWeighted(src1=line_img, alpha=1., src2=dilation_img, beta=0.3, gamma=0.)
        img[line_img.shape[0]-40:line_img.shape[0], x_average-10:x_average+10] = 150
        img = cv2.line(img, (y_goal_draw, x_goal_draw), (240, 360), 245, 3)
        return img

    def custom_draw_line(self, dilation_img, line_img, m, n, idx):
        a, b = line_img.shape[0], line_img.shape[1]//2
        for i in range(0,120):
            if dilation_img[(line_img.shape[0]-1) +m*i][line_img.shape[1]//2 +n*i] == 0:
                a, b = ((line_img.shape[0]-1) +m*i, line_img.shape[1]//2 +n*i)
            else:
                line_img = cv2.line(line_img, (line_img.shape[1]//2, line_img.shape[0]), (b, a), 245, 3)
                break
        start_point= np.array((line_img.shape[1]//2, line_img.shape[0]))
        detect_point = np.array((b, a))
        dist = np.linalg.norm(start_point - detect_point)
        return [a, b, dist] 

    def imageCb(self,frame):

        frame = self.bridge.imgmsg_to_cv2(frame, "mono8")

        loop_start = time.time()
                
        frame = transform_img({'img': frame})['img']
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = self.visualize(img)

        image_msg = self.bridge.cv2_to_imgmsg(img, "mono8")
        self.pub_viz_.publish(image_msg)

        self.pub_input_.publish(self.input_msg1) #c

if __name__ == "__main__":
    # Initialize node
    rospy.init_node("detector_manager_node")
    dm = DetectorManager()
