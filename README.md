# Udacity Final Project
**System integration for a real self-driving car**

## Introduction
The goal of this project is to integrate the perception, planning and control software subsystems for a provided Udacity car (called "Carla") so that the car can drive around a given test track using a waypoint navigation. Waypoints are an ordered set of coordinates (in a real world or in a simulator). Since we work remotely from different places and the car is located in the Udacity headquarter, we use a simulator during our development which is provided by Udacity. 

The provided car has these hardware specifications:

* 31.4 GiB Memory
* Intel Core i7-6700K CPU @ 4 GHz x 8
* TITAN X Graphics
* 64-bit OS

The perception subsystem contains obstacle and traffic light detection. The detection provides a traffic light color detection so that the car knows when to stop or drive if the car approaches an intersection with a traffic light.

In the planning subsystem, we implement a waypoint updater for updating the next waypoint depending on the desired behavior. The throttle, break, and steering of the car are actuated by the control subsystem. The implemented subsystem overview for this project can be visualized as following:

![subsystem architecture][image1]


## The Team

| Name | Location | Function |
|:---:|:---:|:---:|
|Dongping Xie| Germany | team lead|
|David Browne | South Africa | team member|
|Klemens Esterle  | Germany |team member|
|Martin Kretzer | Germany | team member|
|Yongkie Wiyogo | Germany | team member|

## Getting Started
### Installation 

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop). 
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space
  
  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```

[Optional]: If you are using a VM for Virtual Box and want to run the Simulator on your host system, so you need to manually enable port 4567. Follow the following points: 

   Port Forwarding
   - First open up Oracle VM VirtualBox
   - Click on the default session and select settings.
   - Click on Network, and then Advanced.
   - Click on Port Forwarding
   - Click on the green plus, adds new port forwarding rule.
   - Add a rule that has 4567 as both the host and guest IP.

3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```

[//]: # (Image References)
[image1]: ./imgs/subsystem_arch.png