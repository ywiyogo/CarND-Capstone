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

## Traffic Light Detection

We researched the state-of-the-art of the fast detection algorithm using deep learning approach. We found out several interesting articles and papers about object detection in general and also for our traffic light detection case. We studied these articles and papers:

* [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf)
* [SqueezeDet](https://arxiv.org/pdf/1612.01051.pdf)
* [Faster R-CNN and the anchor concept](https://arxiv.org/pdf/1506.01497.pdf)
* [Single Shot Multibox Detector (SSD)](https://arxiv.org/pdf/1512.02325.pdf)
* [MobileNets](https://arxiv.org/pdf/1704.04861.pdf)
* [D.Brailovsky: The winner of Nexar Traffic Light Challange](https://medium.freecodecamp.org/recognizing-traffic-lights-with-deep-learning-23dae23287cc)
* [A.Sarkis article](https://medium.com/@anthony_sarkis/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58)

We tried SqueezeNet in order detect and classify the traffic light color based on the article from D.Brailovsky above. We tried to train SqeeuzeNet and the LARA dataset. However, we cannot achieve a good result. Our second approach was to implement the SqueezeDet with the Bosch small traffic light dataset, Udacity bag files, and simulator image dataset. Again, we failed to adapt and integrate this model.

Our last approach uses SSD and MobileNets with COCO dataset. The pre-trained TensorFlow model of this approach can be downloaded from this [repository](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). After the detection, we crop the images from the resulting bounding boxes. The light color classification is performed by performing a histogram color matching algorithm.

An example result from the real video of the Udacity car can be seen below:
<p align="center">
![Detection result][gif1]
</p>

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
[gif1]: ./imgs/udacity_test1.gif