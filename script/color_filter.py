#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image



class Filter():
    def __init__(self):
        #self.lower_col = np.array([20, 70, 20])
        #self.upper_col = np.array([100, 255, 100])

        self.lower_col = np.array([0, 0, 0])
        self.upper_col = np.array([100, 255, 100])

        self.publisher = rospy.Publisher('filtered_image', Image, queue_size=5)
        self.subscriber = rospy.Subscriber('head_camera/rgb/image_raw', Image, self.camera_cb)

        self.bridge = CvBridge()



    def camera_cb(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        mask = cv2.inRange(image, self.lower_col, self.upper_col)
        result = cv2.bitwise_and(image, image, mask=mask)
        self.publish_filtered(result, data)

    def publish_filtered(self, image, data):
        new_image = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        data.data = new_image.data
        self.publisher.publish(data)



if __name__=="__main__":
    rospy.init_node("image_filter", anonymous=True)
    filter = Filter()
    rospy.spin()
