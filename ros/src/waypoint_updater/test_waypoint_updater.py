#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import math

from rospy_message_converter.json_message_converter import convert_json_to_ros_message

import waypoint_updater

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache



_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

def test_find_closest_point():

    pose, lane, _ = get_data()

    closest_index_manual = 356  # calculated by hand

    closest_index = waypoint_updater.get_closest_waypoint_index(lane.waypoints, pose)

    assert closest_index == closest_index_manual


def test_constant_velocity_waypoints():

    _, lane, _ = get_data()

    velocity = 30
    incremental = True
    start_index = 20
    length = 10

    waypoints = lane.waypoints[20:start_index+length]

    velocity_waypoints = waypoint_updater.constant_v_waypoints(waypoints,
                                                               velocity,
                                                               incremental)

    assert len(velocity_waypoints) == length - 1
    for waypoint in velocity_waypoints:
        vx = waypoint.twist.twist.linear.x
        vy = waypoint.twist.twist.linear.y
        v_total = math.sqrt(vx**2 + vy**2)
        assert abs(v_total - velocity) < 1e-4


@lru_cache()
def get_data():
    data_directory = os.path.join(_DIRECTORY, 'data')
    pose_file = os.path.join(data_directory, 'pose_sample.json')
    lane_file = os.path.join(data_directory, 'lane_sample.json')
    traffic_lights_file = os.path.join(data_directory, 'sample_traffic_lights.json')

    with open(pose_file) as pose_file_handle:
        pose_json = pose_file_handle.read()

    with open(lane_file) as lane_file_handle:
        lane_json = lane_file_handle.read()

    with open(traffic_lights_file) as traffic_lights_file_handle:
        traffic_lights_json = traffic_lights_file_handle.read()

    pose = convert_json_to_ros_message('geometry_msgs/PoseStamped', pose_json)
    lane = convert_json_to_ros_message('styx_msgs/Lane', lane_json)
    traffic_lights = convert_json_to_ros_message('styx_msgs/TrafficLightArray',
                                                 traffic_lights_json)

    return pose, lane, traffic_lights 
