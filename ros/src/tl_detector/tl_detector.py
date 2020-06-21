#!/usr/bin/env python
import rospy
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
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.base_waypoints = None
        self.base_waypoints_2d = None
        self.base_waypoints_tree = None
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
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()
        # self.loop()
    
    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.image_cb(Image())
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, msg):
        if not self.base_waypoints or not self.base_waypoints_2d:
            self.base_waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in msg.waypoints]
            self.base_waypoints = msg.waypoints
            self.base_waypoints_tree = KDTree(self.base_waypoints_2d)
            rospy.loginfo("base_waypoints received successfully.")
        else:
            rospy.logwarn("base_waypoints has already been received and initialized.")

    def traffic_cb(self, msg):
        self.lights = msg.lights
        rospy.logdebug("number of lights: {}".format(len(self.lights)))

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if self.base_waypoints_tree is None:
            rospy.logwarn("Base waypoints has not been received yet.")
            return
        self.has_image = True
        self.camera_image = msg
        stopline_wp, state = self.process_traffic_lights()

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
            stopline_wp = stopline_wp if state == TrafficLight.RED else -1
            self.last_wp = stopline_wp
            self.upcoming_red_light_pub.publish(Int32(stopline_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1
        rospy.logdebug("last_state: {}, state_count: {}, state: {}, last_wp: {}".format(self.last_state, self.state_count, self.state, self.last_wp))

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.base_waypoints_2d

        """
        _, closest_wp_idx = self.base_waypoints_tree.query(np.array([x, y]), 1)
        return closest_wp_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)
        # return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        closest_stopline_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.position.x, self.pose.position.y)
            min_dist = len(self.base_waypoints) # set to maximum
            # iterate through the ground truth location of all traffic lights
            for i, light in enumerate(self.lights):
                stop_line = stop_line_positions[i]
                stopline_wp_idx = self.get_closest_waypoint(stop_line[0], stop_line[1])
                dist = stopline_wp_idx - car_wp_idx
                if dist >= 0 and dist < min_dist:
                    min_dist = dist
                    closest_light = light
                    closest_stopline_wp_idx = stopline_wp_idx

        #TODO find the closest visible traffic light (if one exists)

        if closest_light:
            state = self.get_light_state(closest_light)
            return closest_stopline_wp_idx, state
        # self.base_waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
