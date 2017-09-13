#!/usr/bin/env python


import rospy

from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray

import scipy.spatial
import math
import numpy as np

LOOKAHEAD_WPS = 30  # 200 requires too much CPU.


def constant_v_waypoints(waypoints, velocity, incremental=True):
    final_waypoints = []

    if incremental:
        forward_velocity = velocity
    else:
        forward_velocity = - velocity


    x = [waypoint.pose.pose.position.x for waypoint in waypoints]
    y = [waypoint.pose.pose.position.y for waypoint in waypoints]
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    delta_s = np.sqrt(np.square(delta_x) + np.square(delta_y))

    last_waypoint = waypoints[0]
    for index, waypoint in enumerate(waypoints[1:]):
        velocity_x = forward_velocity * delta_x[index] / delta_s[index]
        velocity_y = forward_velocity * delta_y[index] / delta_s[index]
        last_waypoint.twist.twist.linear.x = velocity_x
        last_waypoint.twist.twist.linear.y = velocity_y
        final_waypoints.append(last_waypoint)

        last_waypoint = waypoint

    return final_waypoints


def waypoints_under_lights(waypoints, lights, incremental=True):
    look_ahead = 50
    road_width = 30
    length_zero_velocity = 8 

    final_waypoints = []

    if incremental:
        forward_direction = 1
    else:
        forward_direction = - 1

    x = [waypoint.pose.pose.position.x for waypoint in waypoints]
    y = [waypoint.pose.pose.position.y for waypoint in waypoints]
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    delta_s = np.sqrt(np.square(delta_x) + np.square(delta_y))

    last_waypoint = waypoints[0]
    for index, waypoint in enumerate(waypoints[1:]):
        direction_x = forward_direction * delta_x[index] / delta_s[index]
        direction_y = forward_direction * delta_y[index] / delta_s[index]

        longitudinal_road_vector = np.array([direction_x, direction_y])
        lateral_road_vector = np.array([- direction_y, direction_y])

        ratio = 1
        x_waypoint = last_waypoint.pose.pose.position.x
        y_waypoint = last_waypoint.pose.pose.position.y

        for light in lights:
            relative_position = np.array([light.pose.pose.position.x - x_waypoint,
                                          light.pose.pose.position.y - y_waypoint])

            longitudinal_position = relative_position.dot(longitudinal_road_vector)
            lateral_position = relative_position.dot(lateral_road_vector)

            if (longitudinal_position >= 0 and light.state == 0
                and longitudinal_position <= length_zero_velocity
                and abs(lateral_position) <= road_width):
                this_ratio = 0
            elif (longitudinal_position >= 0 and light.state == 0
                  and longitudinal_position <= look_ahead
                  and abs(lateral_position) <= road_width):
                this_ratio = ((longitudinal_position - length_zero_velocity)
                         / (look_ahead - length_zero_velocity))
            else:
                this_ratio = 1

            ratio = min(this_ratio, ratio)
                
        old_velocity_x = last_waypoint.twist.twist.linear.x
        old_velocity_y = last_waypoint.twist.twist.linear.y
        last_waypoint.twist.twist.linear.x = old_velocity_x * ratio
        last_waypoint.twist.twist.linear.y = old_velocity_x * ratio
        final_waypoints.append(last_waypoint)
        last_waypoint = waypoint

    final_waypoints.append(last_waypoint)

    return final_waypoints


def get_closest_waypoint_index(waypoints, pose):

    waypoint_coordinates = [[waypoint.pose.pose.position.x,
                             waypoint.pose.pose.position.y] for waypoint in waypoints]

    pose_coordinates = [pose.pose.position.x, pose.pose.position.y]
    _, index = scipy.spatial.KDTree(waypoint_coordinates).query(pose_coordinates)

    return index


class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')
        # queue_size: after receiving this number of messages, old messages will be deleted
        # queue_size: all py-files seem to have queue_size=1, so I am using 1 here as well
        # exception: the cpp-files in waypoint_follower have queue_size=10
        # todo: test if cpu has less issues with different queue_size and loop-frequency
        self.waypoints = None
        self.lights = []

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        rospy.spin()

    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints

    def pose_cb(self, pose):

        velocity = 20
        if self.waypoints is not None:
            closest_wp_index = get_closest_waypoint_index(self.waypoints, pose)
            waypoints_2laps = self.waypoints + self.waypoints
            lane = Lane()
            velocity_waypoints = constant_v_waypoints(waypoints_2laps[closest_wp_index:
                                                                      closest_wp_index+LOOKAHEAD_WPS],
                                                      velocity)
            lane.waypoints = waypoints_under_lights(velocity_waypoints, self.lights)
            
            self.final_waypoints_pub.publish(lane)


    def traffic_cb(self, traffic_lights):
        self.lights = traffic_lights.lights


    '''
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    '''

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
