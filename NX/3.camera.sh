#!/bin/bash

cd ~/cvbridge_build_ws && source install/setup.bash --extend && roslaunch enet_ros enet_sad_camera.launch
