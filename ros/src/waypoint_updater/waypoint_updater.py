#!/usr/bin/env python

import scipy.spatial
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, TrafficLightArray
from std_msgs.msg import Int32


LOOKAHEAD_WPS = 30


def constant_v_waypoints(waypoints, velocity, incremental=True):
    final_waypoints = []

    if incremental:
        forward_velocity = velocity
    else:
        forward_velocity = - velocity


    x_vector = [waypoint.pose.pose.position.x for waypoint in waypoints]
    y_vector = [waypoint.pose.pose.position.y for waypoint in waypoints]

    delta_x = np.diff(x_vector)
    delta_y = np.diff(y_vector)
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


def waypoints_under_stoppage_point(waypoints, stoppage_point,
                                   incremental=True):
    if stoppage_point is None:
        return waypoints

    look_ahead = 50
    road_width = 30
    start_zero_velocity = 25
    end_zero_velocity = 28

    final_waypoints = []

    if incremental:
        forward_direction = 1
    else:
        forward_direction = - 1

    x_vector = [waypoint.pose.pose.position.x for waypoint in waypoints]
    y_vector = [waypoint.pose.pose.position.y for waypoint in waypoints]
    delta_x = np.diff(x_vector)
    delta_y = np.diff(y_vector)
    delta_s = np.sqrt(np.square(delta_x) + np.square(delta_y))

    last_waypoint = waypoints[0]

    for index, waypoint in enumerate(waypoints[1:]):
        direction_x = forward_direction * delta_x[index] / delta_s[index]
        direction_y = forward_direction * delta_y[index] / delta_s[index]

        longitudinal_road_vector = np.array([direction_x, direction_y])
        lateral_road_vector = np.array([-direction_y, direction_x])

        x_waypoint = last_waypoint.pose.pose.position.x
        y_waypoint = last_waypoint.pose.pose.position.y

        relative_position = np.array([stoppage_point.pose.pose.position.x - x_waypoint,
                                      stoppage_point.pose.pose.position.y - y_waypoint])

        longitudinal_position = relative_position.dot(longitudinal_road_vector)
        lateral_position = relative_position.dot(lateral_road_vector)

        if (longitudinal_position >= start_zero_velocity
                and longitudinal_position <= end_zero_velocity
                and abs(lateral_position) <= road_width):
            ratio = 0
        elif (longitudinal_position >= end_zero_velocity
              and longitudinal_position <= look_ahead
              and abs(lateral_position) <= road_width):
            ratio = ((longitudinal_position - end_zero_velocity)
                     / (look_ahead - end_zero_velocity))
        else:
            ratio = 1


        last_waypoint.twist.twist.linear.x *= ratio
        last_waypoint.twist.twist.linear.y *= ratio
        final_waypoints.append(last_waypoint)
        last_waypoint = waypoint

    final_waypoints.append(last_waypoint)

    return final_waypoints


def waypoints_under_lights(waypoints, lights, incremental=True):
    red = 0

    look_ahead = 50
    road_width = 30
    length_zero_velocity = 15

    final_waypoints = []

    if incremental:
        forward_direction = 1
    else:
        forward_direction = - 1

    x_vector = [waypoint.pose.pose.position.x for waypoint in waypoints]
    y_vector = [waypoint.pose.pose.position.y for waypoint in waypoints]
    delta_x = np.diff(x_vector)
    delta_y = np.diff(y_vector)
    delta_s = np.sqrt(np.square(delta_x) + np.square(delta_y))

    last_waypoint = waypoints[0]
    for index, waypoint in enumerate(waypoints[1:]):
        direction_x = forward_direction * delta_x[index] / delta_s[index]
        direction_y = forward_direction * delta_y[index] / delta_s[index]

        longitudinal_road_vector = np.array([direction_x, direction_y])
        lateral_road_vector = np.array([-direction_y, direction_y])

        ratio = 1
        x_waypoint = last_waypoint.pose.pose.position.x
        y_waypoint = last_waypoint.pose.pose.position.y

        for light in lights:
            relative_position = np.array([light.pose.pose.position.x - x_waypoint,
                                          light.pose.pose.position.y - y_waypoint])

            longitudinal_position = relative_position.dot(longitudinal_road_vector)
            lateral_position = relative_position.dot(lateral_road_vector)

            if (longitudinal_position >= 0 and light.state == red
                    and longitudinal_position <= length_zero_velocity
                    and abs(lateral_position) <= road_width):
                ratio = 0
            elif (longitudinal_position >= 0 and light.state == red
                  and longitudinal_position <= look_ahead
                  and abs(lateral_position) <= road_width):
                ratio = ((longitudinal_position - length_zero_velocity)
                         / (look_ahead - length_zero_velocity))
            else:
                ratio = 1

        last_waypoint.twist.twist.linear.x *= ratio
        last_waypoint.twist.twist.linear.y *= ratio
        final_waypoints.append(last_waypoint)
        last_waypoint = waypoint

    final_waypoints.append(last_waypoint)

    return final_waypoints


def get_kd_tree(waypoints):
    if get_kd_tree.kd_tree is None:
        waypoint_coordinates = [[waypoint.pose.pose.position.x,
                                 waypoint.pose.pose.position.y] for waypoint in waypoints]
        get_kd_tree.kd_tree = scipy.spatial.KDTree(waypoint_coordinates)
        get_kd_tree.waypoint_coordinates = waypoint_coordinates
        return get_kd_tree.kd_tree, get_kd_tree.waypoint_coordinates
    else:
        return get_kd_tree.kd_tree, get_kd_tree.waypoint_coordinates

get_kd_tree.kd_tree = None
get_kd_tree.waypoint_coordinates = None


def get_closest_index_behind(waypoints, pose, incremental=True):

    waypoints_kd_tree, waypoint_coordinates = get_kd_tree(waypoints)

    pose_coordinates = [pose.pose.position.x, pose.pose.position.y]
    _, index = waypoints_kd_tree.query(pose_coordinates)

    next_index = (index + 1) % len(waypoints)
    this_position = np.array(waypoint_coordinates[index])
    next_position = np.array(waypoint_coordinates[next_index])
    positive_vector = (this_position - next_position if incremental
                       else this_position - next_position)
    relative_position = np.array(pose_coordinates) - this_position
    if (positive_vector.dot(relative_position) >= 0):
        return index
    else:
        adjusted_index = index - 1 if incremental else index + 1
        return adjusted_index % len(waypoints)


class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.waypoints = None
        self.waypoints_2laps = None
        self.lights = []
        self.stoppage_point = None

        self.waypoints_subscriber = rospy.Subscriber('/base_waypoints', Lane,
                                                     self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.next_light_cb, queue_size=1)
        self.velocity = rospy.get_param('/waypoint_loader/velocity', 20) / 3.6

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        rospy.spin()


    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints
        self.waypoints_2laps = self.waypoints + self.waypoints
        self.waypoints_subscriber.unregister()


    def next_light_cb(self, next_light_index_message):
        next_light_index = next_light_index_message.data
        if next_light_index == -1:
            self.stoppage_point = None
        else:
            next_light = self.waypoints[next_light_index]
            self.stoppage_point = next_light


    def pose_cb(self, pose):
        velocity = 10
        if self.waypoints is not None:
            closest_wp_index = get_closest_index_behind(self.waypoints, pose)
            lane = Lane()
            next_points = self.waypoints_2laps[closest_wp_index:
                                               closest_wp_index+LOOKAHEAD_WPS]
            velocity_waypoints = constant_v_waypoints(next_points, self.velocity)
            lane.waypoints = waypoints_under_stoppage_point(velocity_waypoints,
                                                            self.stoppage_point)

            self.final_waypoints_pub.publish(lane)


class WaypointUpdaterGroundTruth(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.waypoints = None
        self.waypoints_2laps = None
        self.lights = []

        self.waypoints_subscriber = rospy.Subscriber('/base_waypoints', Lane,
                                                     self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                         self.traffic_cb, queue_size=1)
        self.velocity = rospy.get_param('/waypoint_loader/velocity', 20) / 3.6

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        rospy.spin()


    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints
        self.waypoints_2laps = self.waypoints + self.waypoints
        self.waypoints_subscriber.unregister()


    def pose_cb(self, pose):

        if self.waypoints is not None:
            closest_wp_index = get_closest_index_behind(self.waypoints, pose)
            lane = Lane()
            next_points = self.waypoints_2laps[closest_wp_index:
                                               closest_wp_index+LOOKAHEAD_WPS]
            velocity_waypoints = constant_v_waypoints(next_points, self.velocity)
            lane.waypoints = waypoints_under_lights(velocity_waypoints, self.lights)

            self.final_waypoints_pub.publish(lane)


    def traffic_cb(self, traffic_lights):
        self.lights = traffic_lights.lights


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
