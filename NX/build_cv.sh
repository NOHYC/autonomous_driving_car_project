#!/bin/bash
sudo apt-get install python3-pip python-catkin-tools python3-dev python3-numpy
sudo pip3 install rospkg catkin_pkg
mkdir -p ~/cvbridge_build_ws/src
cd ~/cvbridge_build_ws/src && git clone -b noetic https://github.com/ros-perception/vision_opencv.git
find ~/cvbridge_build_ws/src/vision_opencv/cv_bridge/ -name 'CMakeLists.txt' -exec sed -n -e 's/PYTHON37/PYTHON3/g' CMakeLists.txt
cd ~/cvbridge_build_ws && catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so
catkin config --install && catkin build cv_bridge
source install/setup.bash --extend