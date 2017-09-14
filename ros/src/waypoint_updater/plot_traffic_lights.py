#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from rospy_message_converter.json_message_converter import convert_json_to_ros_message

import numpy as np
import quaternion

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

def plot_and_save_traffic_lights():
    data_directory = os.path.join(_DIRECTORY, 'data')
    traffic_lights_file = os.path.join(data_directory, 'sample_traffic_lights.json')

    with open(traffic_lights_file) as traffic_lights_file_handle:
        traffic_lights_json = traffic_lights_file_handle.read()
    traffic_lights = convert_json_to_ros_message('styx_msgs/TrafficLightArray',
                                                 traffic_lights_json)

    x, y, u, v = [], [], [], []
    for light in traffic_lights.lights:
        x.append(light.pose.pose.position.x)
        y.append(light.pose.pose.position.y)
        v_x = np.array([1, 0, 0])
        rotation = np.quaternion(light.pose.pose.orientation.w,
                                 light.pose.pose.orientation.x,
                                 light.pose.pose.orientation.y,
                                 light.pose.pose.orientation.z)
        v_world = quaternion.as_rotation_matrix(rotation).dot(v_x)
        u.append(v_world[0])
        v.append(v_world[1])

    m = np.hypot(u, v)
    plt.quiver(x, y, u, v, m)
    plt.savefig('result.png')


if __name__ == "__main__":
    plot_and_save_traffic_lights()
