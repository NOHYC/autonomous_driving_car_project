cmake_minimum_required(VERSION 3.0.2)
project(ros_enet)

find_package(catkin REQUIRED
  COMPONENTS
    geometry_msgs
    sensor_msgs
    std_msgs
    message_generation)

#add_message_files(
#  FILES
#    BoundingBox.msg
#    BoundingBoxes.msg
#)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
    std_msgs
)

catkin_package()

include_directories()
