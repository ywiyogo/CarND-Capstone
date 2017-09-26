#!/usr/bin/env python


import waypoint_updater

if __name__ == '__main__':
    try:
        waypoint_updater.WaypointUpdaterGroundTruth()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
