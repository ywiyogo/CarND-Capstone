#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import math

import numpy as np

from rospy_message_converter.json_message_converter import convert_json_to_ros_message

from twist_controller import Controller

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

def start_controller():
    controller = Controller(controller_rate=30,
			    accel_limit=10,
			    decel_limit=-10,
			    max_steer_angle=0.25,
			    vehicle_mass=1000,
			    wheel_radius=0.3)
    return controller
