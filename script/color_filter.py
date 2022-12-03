#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from sensor_msgs import point_cloud2



class Filter():
    def __init__(self):

        #INFO: These two arrays are the HSV value bounds. I used an online
        #      color picker to figure them out, but feel free to adjust if needed
        self.lower_col = np.array([40, 30, 30])
        self.upper_col = np.array([80, 255, 255])

        self.publisher = rospy.Publisher('filtered_image', Image, queue_size=5)
        self.subscriber = rospy.Subscriber('head_camera/rgb/image_raw', Image, self.camera_cb)
        self.depth_subscriber = rospy.Subscriber('head_camera/depth_registered/points', PointCloud2, self.depth_cb)
        self.target_publisher = rospy.Publisher('arm_target_point', PointStamped, queue_size=5)


        self.x = None
        self.y = None


        self.bridge = CvBridge()

        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.minThreshold = 5
        detector_params.maxThreshold = 255
        detector_params.filterByInertia = False
        detector_params.filterByConvexity = False
        detector_params.filterByArea = False
        detector_params.blobColor = 255

        self.detector = cv2.SimpleBlobDetector_create(detector_params)



    def camera_cb(self, data):
        #INFO: If the colors are weird, this may need to be changed to BGR instead of RGB.
        #      There's only a handful of instances of this, so you should be able to find
        #      them without too much trouble with ctrl-F
        image = cv2.cvtColor(self.bridge.imgmsg_to_cv2(data, 'rgb8'), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(image, self.lower_col, self.upper_col)
        result = cv2.bitwise_and(image, image, mask=mask)
        keypoints, result = self.get_keypoints(result)

        #INFO: This filtered image should have a blue circle on/around the largest blob
        self.publish_filtered(result, data)

    def depth_cb(self, data):
        if self.x == None or self.y == None:
            return
        else:
            cloud_gen = point_cloud2.read_points(data)

            #INFO: This is probably insanely slow but should still be fast enough
            cloud_list = [p for p in cloud_gen]
            target_point = cloud_list[(self.y - 1) * 640 + self.x - 1]
            new_point = PointStamped()

            #INFO: The specifics of which is which here might need changing
            new_point.point.x = target_point[0]
            new_point.point.y = target_point[1]
            new_point.point.z = target_point[2]

            #INFO: The point should be in the same frame as the pointcloud.
            #      I don't know how that works in relation to using it as a target point.
            new_point.header.frame_id = data.header.frame_id
            self.target_publisher.publish(new_point)



    def publish_filtered(self, image, data):
        new_image = self.bridge.cv2_to_imgmsg(image, 'rgb8')
        data.data = new_image.data
        self.publisher.publish(data)

    def get_keypoints(self, im):
        #INFO: This should work fine. I'd be pretty concerned if it didn't.
        gray = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 5, 255, cv2.THRESH_BINARY)[1]
        keypoints = self.detector.detect(gray)
        if len(keypoints) > 0:
            keypoints = list(keypoints)
            keypoints.sort(key=lambda x: x.size, reverse=True)
            self.x = int(keypoints[0].pt[0])
            self.y = int(keypoints[0].pt[1])
            im = cv2.drawKeypoints(im, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            print("No keypoints found")
        return keypoints, im



if __name__=="__main__":
    rospy.init_node("image_filter", anonymous=True)
    filter = Filter()
    rospy.spin()
