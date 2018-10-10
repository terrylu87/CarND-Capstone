#!/usr/bin/env python
import os
import glob
import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3
CLASSIFIER_COUNT_THRESHOLD = 3
MAX_TL_WPS = 100 # We don't look for traffic light beyod these many waypoints 

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.classifier_count = CLASSIFIER_COUNT_THRESHOLD
        self.classifier_last_state = TrafficLight.UNKNOWN
        self.classifier_last_wp = -1

        # Create a Traffic light classifier if configured
        if self.config['use_classifier']:
            frozen_model_file = self.get_model_path()
            self.light_classifier = TLClassifier(frozen_model_file)
            # First image classification takes longer, so we classify
            # a dummy image to speed up actual classification later
            width = self.config['camera_info']['image_width']
            height = self.config['camera_info']['image_height']
            dummy_image = np.zeros(shape=(height, width, 3))
            self.light_classifier.get_classification(dummy_image)

        rospy.loginfo('Initialized tl_detector')

        rospy.spin()

    def get_model_path(self):
        model_dir = None
        if self.config['is_site']:
            rospy.loginfo('Loading model for site mode.')
            model_dir = "./model/site"
        else:
            rospy.loginfo('Loading model for sim mode.')
            model_dir = "./model/sim"

        model_file = model_dir + "/frozen_inference_graph.pb"
        if not os.path.exists(model_file):
            with open(model_file, "wb") as model_out:
                for chunk_file in sorted(glob.glob(model_dir + "/model_chunk_*")):
                    with open(chunk_file, "rb") as chunk_in:
                        model_out.write(chunk_in.read())
                model_out.close()
        assert os.path.exists(model_file)
        return model_file
            

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        # Classify every CLASSIFIER_COUNT_THRESHOLDth camera images (Ex. every third or fourth camera image)
        # Classifier will run at the startup time since
        #     self.classifier_count is initialized with CLASSIFIER_COUNT_THRESHOLD
        # This can help with the latency when turning on camera images in the simulator
        if self.classifier_count >= CLASSIFIER_COUNT_THRESHOLD:
            self.has_image = True
            self.camera_image = msg
            light_wp, state = self.process_traffic_lights()
            
            self.classifier_last_wp = light_wp
            self.classifier_last_state = state
            self.classifier_count = 0
        else:
            light_wp = self.classifier_last_wp
            state = self.classifier_last_state
            self.classifier_count += 1

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.config['use_classifier']):
           return light.state

        if (not self.has_image):
           self.prev_light_loc = None
           return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        # find the closest visible traffic light (if one exists)
        diff = MAX_TL_WPS
        for i, temp_light in enumerate(self.lights):
            # Get the stop line index 
            line = stop_line_positions[i]
            line_position = self.get_closest_waypoint(line[0], line[1])
            # Find closest stop line waypoint index
            d = line_position - car_position
            if d > 0 and d < diff:
                diff = d
                light = temp_light
                light_wp = line_position 

        if light:
            state = self.get_light_state(light)
            return light_wp, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
