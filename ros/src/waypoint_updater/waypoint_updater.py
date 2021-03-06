#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level = rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.decel_limit = rospy.get_param('~decel_limit', -5)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.pose = None
        self.base_waypoints = None
        self.base_waypoints_2d = None
        self.base_waypoints_tree = None
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.stopline_wp_idx = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # print rospy.Time.now().to_sec()
            if self.pose and self.base_waypoints_2d and self.base_waypoints_tree and self.stopline_wp_idx:
                self.final_waypoints_pub.publish(self.generate_lane())
            rate.sleep()

    def generate_lane(self):
        lane = Lane()

        closest_wp_idx = self.get_closest_wp_idx()
        farthest_wp_idx = closest_wp_idx + LOOKAHEAD_WPS
        initial_wps = self.base_waypoints.waypoints[closest_wp_idx: farthest_wp_idx]

        rospy.loginfo("stopline_wp_idx = {}".format(self.stopline_wp_idx))
        if self.stopline_wp_idx == -1 or self.stopline_wp_idx > farthest_wp_idx:
            lane.waypoints = initial_wps
        else:
            lane.waypoints = self.decelerate_wps(initial_wps, closest_wp_idx)
        
        return lane
    
    def decelerate_wps(self, initial_wps, closest_wp_idx):
        waypoints = []
        for i, wp in enumerate(initial_wps):
            p = Waypoint()
            p.pose = wp.pose
            stop_idx = max(self.stopline_wp_idx - closest_wp_idx - 2 ,0)
            dist = self.distance(initial_wps, i, stop_idx)
            rospy.logdebug("distance = {}, decel_limit = {}".format(dist, self.decel_limit))
            vel = math.sqrt(2.*math.fabs(self.decel_limit)*dist)
            if vel < 1.:
                vel = 0.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            waypoints.append(p)
        return waypoints
    
    def get_closest_wp_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        _, closest_wp_idx = self.base_waypoints_tree.query(np.array([x, y]), 1)
        prev_wp_idx = closest_wp_idx - 1

        closest_coord = self.base_waypoints_2d[closest_wp_idx]
        prev_coord = self.base_waypoints_2d[closest_wp_idx - 1]

        closest_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        curr_vect = np.array([x,y])

        dot_product = np.dot(curr_vect - closest_vect, closest_vect - prev_vect)

        if dot_product > 0:
            # closest waypoint is behind current pose
            closest_wp_idx = (closest_wp_idx+1) % len(self.base_waypoints_2d)

        return closest_wp_idx

    def pose_cb(self, msg):
        self.pose = msg
        rospy.logdebug("current pose has been received: {},{}".format(msg.pose.position.x, msg.pose.position.y))

    def waypoints_cb(self, waypoints):
        if not self.base_waypoints or not self.base_waypoints_2d:
            self.base_waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.base_waypoints = waypoints
            self.base_waypoints_tree = KDTree(self.base_waypoints_2d)
            rospy.logdebug("base_waypoints received successfully.")
        else:
            rospy.logwarn("base_waypoints has already been received and initialized.")

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
