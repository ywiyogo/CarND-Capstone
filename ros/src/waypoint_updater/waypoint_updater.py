#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

LOOKAHEAD_WPS = 30 #200 # I am using 30 here because I am using loop-frequency = 30. Thus, it corresponds to the waypoints during the next second. 200 requires too much CPU.


class WaypointUpdater(object):
    
    def __init__(self):
        rospy.init_node('waypoint_updater')
        # queue_size: after receiving this number of messages, old messages will be deleted
        # queue_size: all py-files seem to have queue_size=1, so I am using 1 here as well
        # exception: the cpp-files in waypoint_follower have queue_size=10
        # todo: test if cpu has less issues with different queue_size and loop-frequency
        self.waypoints = None
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1) # note: cpp-file really takes final_waypoints without backslash!
        rospy.spin()

    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints

    def pose_cb(self, msg):
        if self.waypoints is not None:
            pose = msg.pose
            # iterate through list of waypoints and get the waypoint which is closest to my current position:
            closest_wp_index = 0
            closest_wp_dist = 1000000.1
            for i, wp in enumerate(self.waypoints):
                wp_dist = math.sqrt( (pose.position.x-wp.pose.pose.position.x)**2 + (pose.position.y-wp.pose.pose.position.y)**2 )
                if wp_dist < closest_wp_dist:
                    closest_wp_index = i
                    closest_wp_dist = wp_dist

            # after the last waypoint in the waypoint list, the first waypoint will follow again. Now a list for 2 laps:
            waypoints_2laps = self.waypoints + self.waypoints
            lane = Lane()
            lane.waypoints = waypoints_2laps[ closest_wp_index : closest_wp_index+LOOKAHEAD_WPS ] # get waypoints between closest wp and the last wp that should be predicted
            self.final_waypoints_pub.publish(lane)

    
''' Functions not used yet:
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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
