#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
import math

from twist_controller import Controller

import pid

class DBWNode(object):
    
    def __init__(self):
        rospy.init_node('dbw_node')

        # Params:
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        # Subscriber:
        self.dbw_enable = False
        self.twist_cmd = None
        self.current_velocity = None
        self.current_pose = None
        rospy.Subscriber('/twist_cmd',TwistStamped,self.twist_cmd_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enable',Bool,self.dbw_enable_cb)

        # Controller:
        self.controller = Controller(accel_limit, decel_limit, max_steer_angle)

        # Publisher:
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',BrakeCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',SteeringCmd, queue_size=1)

        # Execute loop for each message (this depends on the defined rate):
        self.loop()

    def twist_cmd_cb(self, msg):
        self.twist_cmd = msg.twist

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist

    def current_pose_cb(self, msg):
        self.current_pose = msg.pose

    def dbw_enable_cb(self, msg):
        self.dbw_enable = bool(msg.data)
        if self.dbw_enable is False:
            self.controller.reset()

    def loop(self):
        rate = rospy.Rate(30) # 30 is defined in cpp-file as loop-frequency # 50Hz
        while not rospy.is_shutdown():
            if self.dbw_enable:
                # throttle, brake, steer = self.controller.control(<proposed linear velocity>,
                #                                                     <proposed angular velocity>,
                #                                                     <current linear velocity>,
                #                                                     <dbw status>, --> already checked!
                #                                                     <any other argument you need>)
                proposed_linear_velocity = 0.0
                proposed_angular_velocity = 0.0
                current_linear_velocity = 0.0

                # Calculate errors
                cross_track_error = 0 # TODO: calculate cte
                speed_error = proposed_linear_velocity - current_linear_velocity

                throttle, brake, steer = self.controller.control(cross_track_error, speed_error)
                
                # Publisher:
                # self.publish(5.,0.,10.) # for testing purposes
                self.publish(throttle, brake, steer)

            rate.sleep() # wiki.ros.org/rospy/Overview/Time#Sleeping_and_Rates --> wait until next rate

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)
    
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        
if __name__ == '__main__':
    DBWNode()
