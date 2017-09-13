#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import math

import numpy as np

from rospy_message_converter.json_message_converter import convert_json_to_ros_message
from geometry_msgs.msg import TwistStamped, PoseStamped

from twist_controller import Controller

from styx_msgs.msg import Lane
from styx_msgs.msg import Waypoint

from dbw_common import get_cross_track_error_from_frenet

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache



_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

def test_control_step():

    controller = start_controller()

    throttle_out, brake_out, steering_out = controller.control(1,0.1)

    assert throttle_out > 0

def test_control_speed_ramp():

    controller = start_controller()

    velocity_error = np.arange(0,10,0.1)
    steering_error = np.zeros(100)
    assert len(velocity_error) == len(steering_error)

    for idx, vel_err in enumerate(velocity_error):
        throttle_out, brake_out, steering_out = controller.control(vel_err, steering_error[idx])

    assert throttle_out == 1 and brake_out == 0 and steering_out == 0

def test_control_idle():

    controller = start_controller()

    velocity_error = np.zeros(100)
    steering_error = np.zeros(100)
    assert len(velocity_error) == len(steering_error)
    
    for idx, vel_err in enumerate(velocity_error):
        throttle_out, brake_out, steering_out = controller.control(vel_err, steering_error[idx])

    assert throttle_out == 0 and brake_out == 0 and steering_out == 0

def test_lane_boundary():

    controller = start_controller()

    velocity_error = np.zeros(100)
    steering_error = np.flipud(np.arange(1,2,0.01))
    assert len(velocity_error) == len(steering_error)
    
    for idx, vel_err in enumerate(velocity_error):
        throttle_out, brake_out, steering_out = controller.control(vel_err, steering_error[idx])

    assert throttle_out == 0 and brake_out == 0 and steering_out > 0

def test_get_cross_track_error_from_frenet():
    TestWaypoints = Lane()

    xrange = np.arange(0, 100, 0.01)
    yrange = np.arange(0, 100, 0.01)
    currentPose = PoseStamped()
    currentPose.pose.position.x = 20
    currentPose.pose.position.y = 10

    for idx,xval in enumerate(xrange):
        wp = Waypoint()
        wp.pose.pose.position.x = xrange[idx]
        wp.pose.pose.position.y = yrange[idx]
        wp.twist.twist.linear.x = 20 # longitudinal velocity
        wp.twist.twist.linear.y = 0 # lateral velocity
        TestWaypoints.waypoints.append(wp)

    cte = get_cross_track_error_from_frenet(TestWaypoints.waypoints, currentPose.pose)

    assert cte == -10*np.cos(math.pi/4)

def start_controller():
    controller = Controller(controller_rate=30,
			    accel_limit=10,
			    decel_limit=-10,
			    max_steer_angle=0.25,
			    vehicle_mass=1000,
			    wheel_radius=0.3)
    return controller