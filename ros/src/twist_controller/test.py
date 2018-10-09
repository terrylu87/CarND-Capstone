#!/usr/bin/env python

import rospy
from tf.msg import tfMessage
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TwistStamped, PoseStamped, Vector3
from styx_msgs.msg import Lane, Waypoint
from math import atan2

ANGLES = tuple
heading = 0


def pose_cb(msg):
    global ANGLES
    q = msg.pose.orientation
    ANGLES = euler_from_quaternion([q.x, q.y, q.z, q.w])


def wp_cb(waypoints):
    global ANGLES
    global heading
    first = waypoints.waypoints[0]
    try:
        second = waypoints.waypoints[1]
    except:
        # End of path
        pass
    else:
        diffx = second.pose.pose.position.x - first.pose.pose.position.x
        diffy = second.pose.pose.position.y - first.pose.pose.position.y
        heading = atan2(diffy, diffx)
        print heading
        print ANGLES


if __name__ == '__main__':

    rospy.init_node('test_node')

    rospy.Subscriber('/current_pose', PoseStamped, pose_cb)
    rospy.Subscriber('/final_waypoints', Lane, wp_cb)

    rospy.spin()
