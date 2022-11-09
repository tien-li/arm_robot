import rospy
from apriltag_ros.msg import AprilTagDetectionArray

class AprilTagListener:
    def __init__(self):
        self.image_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.callback)

    def callback(self, data):
        self.apriltag_data = data
        self.apriltag_data.detections = sorted(self.apriltag_data.detections, key=lambda a: a.id[0])
