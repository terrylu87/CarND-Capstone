#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Float32
import rospy
from tf.msg import tfMessage
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped, Vector3, Point
from styx_msgs.msg import Lane, Waypoint
import math
import numpy as np

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

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

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)
        self.target_heading_pub = rospy.Publisher('/target_heading', Float32, queue_size=1)
        self.current_heading_pub = rospy.Publisher('/current_heading', Float32, queue_size=1)
        self.command_angle_pub = rospy.Publisher('/command_angle', Float32, queue_size=1)

        # TODO: Create `Controller` object
        self.controller = Controller(vehicle_mass=vehicle_mass,
                                     fuel_capacity=fuel_capacity,
                                     brake_deadband=brake_deadband,
                                     decel_limit=decel_limit,
                                     accel_limit=accel_limit,
                                     wheel_radius=wheel_radius,
                                     wheel_base=wheel_base,
                                     steer_ratio=steer_ratio,
                                     max_lat_accel=max_lat_accel,
                                     max_steer_angle=max_steer_angle)

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/final_waypoints', Lane, self.wp_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        self.current_vel = None
        self.dbw_enabled = None
        self.linear_vel = None
        self.angular_vel = None
        self.throttle = self.steering = self.brake = 0
        self.target_heading = 0.
        self.current_heading = 0.
        self.command_angle = 0.
        self.position = None
        self.rel_distance = 10
        self.steer_ratio = steer_ratio

        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            if not None in (self.current_vel, self.linear_vel, self.angular_vel):
                self.throttle, self.brake, self.steering = \
                        self.controller.control(self.current_vel, self.dbw_enabled,
                                                self.linear_vel, self.current_heading, self.command_angle)

            if self.dbw_enabled:
                self.publish(self.throttle, self.brake, self.steering)
            rate.sleep()

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg

    def twist_cb(self, msg):
        self.linear_vel = msg.twist.linear.x
        self.angular_vel = msg.twist.angular.z

    def velocity_cb(self, msg):
        self.current_vel = msg.twist.linear.x

    def wp_cb(self, waypoints):
        first = waypoints.waypoints[0]
        try:
            second = waypoints.waypoints[5]
        except:
            # End of path
            pass
        else:
            diffx = second.pose.pose.position.x - first.pose.pose.position.x
            diffy = second.pose.pose.position.y - first.pose.pose.position.y
            self.target_heading = math.atan2(diffy, diffx)
            self.position = first.pose.pose.position

    def distance(self, p1, p2):
        x, y = p1.x - p2.x, p1.y - p2.y
        return math.sqrt(x*x + y*y)
    
    def vector(self, p1, p2):
        return np.array([p2.x - p1.x, p2.y - p1.y])

    def pose_cb(self, msg):
        q = msg.pose.orientation
        p = msg.pose.position
        angles = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_heading = angles[2]
        if self.position:
            offset = self.distance(p, self.position)
            car_wp = self.vector(p, self.position)
            y = 10*math.sin(self.current_heading)
            x = 10*math.cos(self.current_heading)
            car = Point(p.x,p.y,p.z)
            p.x = p.x + x
            p.y = p.y + y
            car_heading = self.vector(car, p)
            cross = np.cross(car_heading, car_wp)
            correction = math.atan(offset/self.rel_distance)*cross/(abs(cross))
            self.command_angle = self.target_heading + correction
        # self.current_heading_pub.publish(angles[2])

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

        self.target_heading_pub.publish(self.target_heading)
        self.current_heading_pub.publish(self.current_heading)
        self.command_angle_pub.publish(self.steering)


if __name__ == '__main__':
    DBWNode()
