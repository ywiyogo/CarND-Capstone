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
from scipy import spatial

STATE_COUNT_THRESHOLD = 3
GET_TRAINING_DATA = False		# Set to True if you want to save training data

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.cust_waypoints = []
        self.cust_tlights = []
        self.detected_tlight = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
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

        if GET_TRAINING_DATA:
            self.approach = False
            self.last_dist = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        #print("[TLD] pose attr: ", dir(self.pose))
        #print("[TLD] pose: ", self.pose)
        #Test
        #print("pose: ", self.pose.position.x, self.pose.position.y)



    def waypoints_cb(self, waypoints):
        if not self.waypoints:
            self.waypoints = waypoints
            # print("[TLD]waypoint dir: \n", dir(waypoints)) # print all attribute of the class
            # print("[TLD]waypoint len: ", len(waypoints.waypoints))
            # print("[TLD]waypoint wp: ", waypoints.waypoints[0])
            # print(" pose attr:", dir(waypoints.waypoints[0].pose.pose.position))
            # print("[TLD]waypoint position: ", waypoints.waypoints[0].pose.pose.position.x)
            for i in range(0, len(self.waypoints.waypoints)):
                self.cust_waypoints.append([self.waypoints.waypoints[i].pose.pose.position.x, self.waypoints.waypoints[i].pose.pose.position.y])


    def traffic_cb(self, msg):
        if not self.lights:
            self.lights = msg.lights
            #print("[TLD] TL: ", dir(self.lights))
            #print(self.lights[0])
            for i in range(0, len(self.lights)):
                tl_pose=[self.lights[i].pose.pose.position.x,self.lights[i].pose.pose.position.y]
                self.cust_tlights.append(tl_pose)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

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

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # YWiyogo: this is a nearest neighbours search problem, not a closest pair of points problem
        # Sort waypoint

        # Divide the x_n points and compare the distance to the x_(n/2) & x_(n/2)+1
        if self.waypoints and pose:
            # tree = KDTree(X, leaf_size=2)
            cust_pose = [pose.position.x, pose.position.y]
            # dist, ind = tree.query(cust_pose, k=1)
            dist,ind = spatial.KDTree(self.cust_waypoints).query(cust_pose)
            #print("[TLD]pose: \n", pose)
            if(ind > len(self.waypoints.waypoints) or ind <0):
                print("[TDL]Err index out of range %d" % ind)
            else:
                print("[TLD] Closest index %d, distance to wp: %f, x: %f, y: %f" % (
                ind, dist, self.waypoints.waypoints[ind].pose.pose.position.x, self.waypoints.waypoints[ind].pose.pose.position.y))
            return ind
        else:
            return -1


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

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

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Commented out for testing...
        #x, y = self.project_to_image_plane(light.pose.pose.position)

        # Get training data
        if GET_TRAINING_DATA:
            self.get_training_data(cv_image)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        enable_imshow= False    #activate to see the camera image
        if enable_imshow:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, desired_encoding="passthrough")
            cv2.imshow("Image window", cv_image)
            cv2.waitKey(1)

        light = None
        light_positions = self.config['light_positions']
        # NOTE, YW: the above light_positions is based on the sim_traffic_light_config.xml
        # but the entries are not the same as from the traffic_cb function !!
        # Currently I concern only the traffic_cb
        #print("light pos: ", light_positions)

        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

            #YW: find the closest visible traffic light (if one exists)
            cust_pose = [self.pose.pose.position.x, self.pose.pose.position.y]
            dist,ind = spatial.KDTree(self.cust_tlights).query(cust_pose)

            diff_x = self.cust_tlights[ind][0] - self.pose.pose.position.x
            if(dist < 70):
                if diff_x >0:
                    if self.detected_tlight != self.cust_tlights[ind]:
                        self.detected_tlight = self.cust_tlights[ind]
                        light = self.lights[ind]
                        print("[TLD] TL %d found, A front distance to current pose: %f" % (ind, dist))


                #else:
                #    print("[TLD] TL is behind the car")


        if light:
            state = self.get_light_state(light)
            return light, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN




    def get_training_data(self, image):
        """Gets training data from the simulator

        """
        # Display camera image
        cv2.imshow("Image window", image)
        cv2.waitKey(3)

        # Variables
        label = 4
        dist_min = 999999999
        dist_threshold = 20000
        light_positions = self.config['light_positions']

        for i, light_pos in enumerate(light_positions):

            delta_x = self.pose.pose.position.x - light_pos[0]
            delta_y = self.pose.pose.position.y - light_pos[1]
            dist = (delta_x * delta_x) + (delta_y * delta_y)

            if dist < dist_min:
                index = i
                dist_min = dist
                sign = delta_x / abs(delta_x)

        """
        index: the index of the closest traffic light
        sign: sign is negative when the vehicle is infront of the nearest traffic light (and visa versa)

        Assign a label to the camera image:
        4 = UNKNOWN
        2 = GREEN
        1 = YELLOW
        0 = RED
        """
        if (dist_min < dist_threshold):

            # Set approach flag if vehicle is infront of the traffic light
            if (self.last_dist - dist_min > 0):
                self.approach = True
            if (self.last_dist - dist_min < 0):
            	self.approach = False

            if (self.approach == True):
                label = self.lights[index].state

        print('Approaching TL? {}, Image Label = {}'.format(self.approach, label))

        # Update previous minimum distance
        self.last_dist = dist_min




if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
