#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh && conda activate base && cd ~/cvbridge_build_ws && source devel/setup.bash && roslaunch ros_enet enet_sad.launch
