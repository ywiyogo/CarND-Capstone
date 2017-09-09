#!/usr/bin/env python

import numpy as np

def get_cross_track_error(waypoints, pose):

    current_pose_x = pose.position.x
    current_pose_y = pose.position.y

    wpts_x = []
    wpts_y = []

    for wp in waypoints:
        wpts_x.append(wp.pose.pose.position.x)
        wpts_y.append(wp.pose.pose.position.y)

    wpts_dist = np.sqrt((np.array(wpts_x) - current_pose_x) ** 2 + (np.array(wpts_y) - current_pose_y) ** 2)

    cte = wpts_dist.min()

    return cte


def test_get_cross_track_error():

    pass


if __name__ == "__main__":
    test_get_cross_track_error()