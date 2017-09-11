#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
#from styx_msgs.msg import Lane

from twist_controller import Controller
from yaw_controller import YawController

from dbw_common import get_cross_track_error, get_cross_track_error_from_frenet

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
        min_speed = 0  # TODO: read from parameter server
        controller_rate = 10  # 30 TODO: read from parameter server, 30 is defined in cpp-file as loop-frequency # 50Hz


        self.controller_rate = controller_rate

        # Subscriber:
        self.dbw_enabled = False
        self.twist_cmd = None
        self.current_velocity = None
        self.current_pose = None
        #self.final_waypoints = None
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        #rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb, queue_size=1)

        # Controller:
        self.speed_and_twist_controller = Controller(controller_rate,
                                                     accel_limit,
                                                     decel_limit,
                                                     max_steer_angle,
                                                     vehicle_mass,
                                                     wheel_radius)
        self.yaw_controller = YawController(wheel_base,
                                            steer_ratio,
                                            min_speed,
                                            max_lat_accel,
                                            max_steer_angle)

        # Publisher:
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)

        # Execute loop for each message (this depends on the defined rate):
        self.loop()

    def twist_cmd_cb(self, msg):
        self.twist_cmd = msg.twist

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist

    def final_waypoints_cb(self, msg):
        self.final_waypoints = msg.waypoints

    def current_pose_cb(self, msg):
        self.current_pose = msg.pose

    def dbw_enabled_cb(self, msg):
        if msg.data:
            self.dbw_enabled = True
        else:
            self.dbw_enabled = False
            self.speed_and_twist_controller.reset()

    def loop(self):
        rate = rospy.Rate(self.controller_rate)
        while not rospy.is_shutdown():
	    rospy.loginfo('dbw_enabled: %s', self.dbw_enabled)
            if self.dbw_enabled:

                proposed_linear_velocity = 10 #self.final_waypoints[0].twist.twist.linear.x
                current_linear_velocity = self.current_velocity.linear.x

                # Calculate errors
                cross_track_error = 0 #get_cross_track_error_from_frenet(self.final_waypoints,self.current_pose)
                speed_error = proposed_linear_velocity - current_linear_velocity
                throttle, brake, steer_twist = self.speed_and_twist_controller.control(speed_error, cross_track_error)

                #linear_velocity = self.twist_cmd.linear.x
                #angular_velocity = self.twist_cmd.angular.z
                #steer_yaw = self.yaw_controller.get_steering(linear_velocity,
                #                                             angular_velocity,
                #                                             current_linear_velocity)
                steer = 0 # steer_twist + steer_yaw

                # Publisher:
                #self.publish(0.5, 0, 0)  # for testing purposes
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
