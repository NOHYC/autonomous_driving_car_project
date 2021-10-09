#!/usr/bin/env python
import rospy

from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDriveStamped

import sys, select, termios, tty

banner = """
Reading from the keyboard  and Publishing to AckermannDriveStamped!
---------------------------
Moving around:
        w
   a    s    d
anything else : stop
CTRL-C to quit
"""

keyBindings = {
  'w':(1,0),
  'd':(1,-1),
  'a':(1,1),
  's':(-1,0)
}

def getKey():
   tty.setraw(sys.stdin.fileno())
   select.select([sys.stdin], [], [], 0)
   key = sys.stdin.read(1)
   termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
   return key

speed = 0.5
turn = 0.25

def vels(speed,turn):
  return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
  settings = termios.tcgetattr(sys.stdin)
  pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=1)
  pub_red = rospy.Publisher('/TFF/RED_Emergency_Stop', Bool, queue_size=1)
  pub_yellow = rospy.Publisher('/TFF/YELLOW_Normal_Stop', Bool, queue_size=1)
  pub_green = rospy.Publisher('/TFF/GREEN_Start_DDPG', Bool, queue_size=1)

  rospy.init_node('keyop')

  x = 0
  th = 0
  status = 0

  print("Keyboard Input ...")

  msg = AckermannDriveStamped();
  msg.header.stamp = rospy.Time.now();
  msg.header.frame_id = "base_link";

  msg.drive.speed = 1;
  msg.drive.acceleration = 1;
  msg.drive.jerk = 1;
  msg.drive.steering_angle = 1
  msg.drive.steering_angle_velocity = 1

  pub.publish(msg)


  try:
    while(1):
       key = getKey()
#       if msg_red.data != True:
#          msg_red.data = True
#          pub_red.publish(msg_red)

#       if msg_yellow.data != True:
#          msg_yellow.data = True
#          pub_yellow.publish(msg_yellow)
           
       if key in keyBindings.keys(): 
          x = keyBindings[key][0]
          th = keyBindings[key][1]
       else:
          x = 0
          th = 0
          if (key == '\x03'):
             break


       if key == 'w' or key == 'd' or key == 'a' or key == 's':  
          msg = AckermannDriveStamped();
          msg.header.stamp = rospy.Time.now();
          msg.header.frame_id = "base_link";

          msg.drive.speed = x*speed;
          msg.drive.acceleration = 1;
          msg.drive.jerk = 1;
          msg.drive.steering_angle = th*turn
          msg.drive.steering_angle_velocity = 1

          pub.publish(msg)

       else:
          if key == 'r':
             msg_red = Bool()
             msg_red.data = False
             pub_red.publish(msg_red)

          elif key == 'y':
             msg_yellow = Bool()
             msg_yellow.data = False
             pub_yellow.publish(msg_yellow)

          elif key == 'g':
             msg_green = Bool()
             msg_green.data = True
             pub_green.publish(msg_green)


  except:
    print 'error'

  finally:
    msg = AckermannDriveStamped();
    msg.header.stamp = rospy.Time.now();
    msg.header.frame_id = "base_link";

    msg.drive.speed = 0;
    msg.drive.acceleration = 1;
    msg.drive.jerk = 1;
    msg.drive.steering_angle = 0
    msg.drive.steering_angle_velocity = 1
    pub.publish(msg)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
