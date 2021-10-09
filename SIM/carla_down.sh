#!/bin/bash

cd ~/wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz && tar -xvf CARLA_0.9.10.1.tar.gz
cd ~/CARLA_0.9.10.1/PythonAPI/carla/dist && sudo python -m easy_install carla-0.9.10-py2.7-linux-x86_64.egg
echo 'source /opt/ros/melodic/setup.bash'>>~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py2.7-linux-x86_64.egg:$CARLA_ROOT/PythonAPI/carla'>>~/.bashrc
source ~/.bashrc

pip install --user pygame numpy
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"

sudo apt-get update 
sudo apt-get install carla-simulator 
cd /opt/carla-simulator
sudo apt-get install carla-simulator=0.9.10-1
