#!/usr/bin/env python

import math
import numpy as np


def get_cross_track_error(waypoints, pose):
    """
    Calculates cross track error from desired path and current pose
    :param waypoints: waypoints (trajectory) vehicle should follow
    :param pose: Current Pose (type PoseStamped) of vehicle
    :return deviation from path formed by waypoints
    """

    wpts_x = []
    wpts_y = []

    current_pose_x = pose.position.x
    current_pose_y = pose.position.y

    for idx, wp in enumerate(waypoints):
        if idx<=10:
            wpts_x.append(wp.pose.pose.position.x)
            wpts_y.append(wp.pose.pose.position.y)

    wpts_dist = np.sqrt((np.array(wpts_x) - current_pose_x) ** 2 + (np.array(wpts_y) - current_pose_y) ** 2)

    cte = wpts_dist.min()

    return cte

def distance(x1, y1, x2, y2):
    return float(np.sqrt((x2-x1)**2 + (y2-y1)**2))

def get_cross_track_error_from_frenet(waypoints, pose):

    n_x = waypoints[1].pose.pose.position.x - waypoints[0].pose.pose.position.x
    n_y = waypoints[1].pose.pose.position.y - waypoints[0].pose.pose.position.y
    x_x = pose.position.x - waypoints[0].pose.pose.position.x
    x_y = pose.position.y - waypoints[0].pose.pose.position.y

    assert ((n_x * n_x + n_y * n_y) != 0);

    # find the projection of x onto n
    proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
    proj_x = proj_norm * n_x
    proj_y = proj_norm * n_y

    frenet_d = distance(x_x, x_y, proj_x, proj_y)

    # see if d value is positive or negative by comparing it to a center point
    center_x = 1e9 - waypoints[0].pose.pose.position.x
    center_y = 1e9 - waypoints[0].pose.pose.position.y
    centerToPos = distance(center_x, center_y, x_x, x_y)
    centerToRef = distance(center_x, center_y, proj_x, proj_y)

    if (centerToPos <= centerToRef):
        frenet_d *= -1;

    return frenet_d


def test_get_cross_track_error():

    pass


if __name__ == "__main__":
    test_get_cross_track_error()
