#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, Point, Pose
from std_msgs.msg import Float32MultiArray #
from geometry_msgs.msg import PoseWithCovarianceStamped

class MovingAverage:
    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n + 1))

    def add_sample(self, new_sample):
        if len(self.data) < self.samples:
            self.data.append(new_sample)
        else:
            self.data = self.data[1:] + [new_sample]
    def get_mm(self):
        return float(sum(self.data)) / len(self.data)
    def get_wmm(self):
        s = 0
        for i, x in enumerate(self.data):
            s += x * self.weights[i]
        return float(s) / sum(self.weights[:len(self.data)])

class Env():
    def __init__(self, action_dim=2):
        self.vehicle_velocity = 0
        self.alpha = 10
        self.beta = 50
        self.gamma = 0.002
        self.cnt = 1
        self.stop_cnt = 0
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('/TFF/cmd_vel', Twist, queue_size=5)
        self.initialpose_publisher = rospy.Publisher(
            "/carla/ego_vehicle/initialpose",
            PoseWithCovarianceStamped, queue_size=1) 
        rospy.Subscriber("/carla/ego_vehicle/imu/imu1",Imu,self.get_imu)
        rospy.Subscriber("/TFF/ddpg_camera_input", Float32MultiArray, self.array_callback)
        
        self.mm_angle = MovingAverage(14)
        self.mm_vel = MovingAverage(14)

        self.past_distance = 0.
        self.action_dim = action_dim

        rospy.on_shutdown(self.shutdown)
        self.laser_input_data = []
        self.imu_data_input = []
        self.data_past = []        


    def get_imu(self,imu_data):
        if len(self.imu_data_input) == 6:
            self.imu_data_input = []
        att_lis = ["angular_velocity","linear_acceleration"] 
        for att1 in att_lis:
            first = "imu_data."+att1
            for att2 in list("xyz"):
                self.imu_data_input.append(eval(first+"."+att2))

    def array_callback(self,data):
        self.laser_input_data = data.data

    
    def shutdown(self):
        rospy.loginfo("Stopping Car")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

#getGoalAngle
    def getGoalDistace(self, input_data):
        x_goal_draw, y_goal_draw = (input_data[0]+input_data[3]+input_data[9]+input_data[12])//4, (input_data[1]+input_data[4]+input_data[10]+input_data[13])//4
        y_goal, x_goal = (input_data[0]+input_data[3]+input_data[9]+input_data[12]-1440)//4, (input_data[1]+input_data[4]+input_data[10]+input_data[13]-960)//4
        theta = math.atan2(-y_goal, x_goal)
        goal_yaw = np.abs(-(theta*57.3-90))
        self.past_distance = goal_yaw
        return goal_yaw

    def getState(self, scan, past_action):
        scan_range = scan
        done = False
        for pa in past_action:
            scan_range.append(pa)
        current_distance = self.getGoalDistace(scan[6:21])
        return scan_range + [current_distance], done


    def setReward(self, state, action, done):
        current_distance = state[-1]
        reward = self.alpha * np.cos(current_distance*np.pi/180) - self.beta * np.abs(np.sin(current_distance*np.pi/180))
        if state[6] == 360 and state[9] == 360 and state[12] == 360 and state[15] == 360 and state[18] == 360:
            done = True
        if action < 1. :
            reward -= 50.
        if done:
            rospy.loginfo("Collision!!")
            reward = -150.
            self.pub_cmd_vel.publish(Twist())
        return reward, done

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]
        linear_param = 2.5
        self.mm_angle.add_sample(ang_vel)
        wmm_angle = self.mm_angle.get_mm()
        self.mm_vel.add_sample(linear_vel)
        wmm_vel = self.mm_vel.get_mm()
        vel_cmd = Twist()
        vel_cmd.linear.x = wmm_vel
        vel_cmd.angular.z = wmm_angle
        self.pub_cmd_vel.publish(vel_cmd)
        data = None
        data1, data2 =[], []

        while data is None:
            try:
                data_imu = rospy.wait_for_message('/carla/ego_vehicle/imu/imu1', Imu, timeout=5)
                data_cam = rospy.wait_for_message('/TFF/ddpg_camera_input', Float32MultiArray, timeout=5)

                att_lis = ["angular_velocity","linear_acceleration"] # ang : gyro , acc : acc
                for att1 in att_lis:
                    first = "data_imu."+att1
                    for att2 in list("xyz"):
                        data1.append(eval(first+"."+att2))
                for att2 in data_cam.data[:]:
                    data2.append(att2)
                data = data1 + data2
            except:
                pass

        if(len(data) != 21):
            data.extend(self.data_past[len(data):])

        self.data_past = data[:]
        state, done = self.getState(data, past_action)
        reward, done = self.setReward(state, wmm_vel * linear_param, done)

        self.cnt += 1
        return np.asarray(state), reward, done

    def reset(self):
        self.cnt = 1
        self.stop_cnt = 0
        self.initial_pose = PoseWithCovarianceStamped()
        self.initial_pose.pose.pose.position.x = 225.252029
        self.initial_pose.pose.pose.position.y = 371.109375
        self.initial_pose.pose.pose.position.z = -2
        self.initialpose_publisher.publish(self.initial_pose)
        vel_cmd = Twist()
        vel_cmd.linear.x = 0
        vel_cmd.angular.z = 0
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        data1, data2 =[], []

        while data is None:
            try:
                data_imu = rospy.wait_for_message('/carla/ego_vehicle/imu/imu1', Imu, timeout=5)
                data_cam = rospy.wait_for_message('/TFF/ddpg_camera_input', Float32MultiArray, timeout=5)
                
                att_lis = ["angular_velocity","linear_acceleration"] # ang : gyro , acc : acc
                for att1 in att_lis:
                    first = "data_imu."+att1
                    for att2 in list("xyz"):
                        data1.append(eval(first+"."+att2))
                for att2 in data_cam.data[:]:
                    data2.append(att2)
                data = data1 + data2
                if data2[0] == 360 and data2[3] == 360 and data2[6] == 360 and data2[9] == 360 and data2[12] == 360:
                    data_cam2 = rospy.wait_for_message('/TFF/ddpg_camera_input', Float32MultiArray, timeout=5)
                    data3 = []
                    for att3 in data_cam2.data[:]:
                        data3.append(att3)
                    data = data1 + data3

            except:
                pass

        self.goal_distance = self.getGoalDistace(data[6:21])
        state, _ = self.getState(data, [0]*self.action_dim)
        return np.asarray(state)
