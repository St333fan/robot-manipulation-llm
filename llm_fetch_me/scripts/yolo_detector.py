#!/usr/bin/env python3

from sensor_msgs.msg import Image
from ultralytics import YOLO, YOLOWorld
import torch
import rospy
import cv2
from custom_msg_srv.srv import InfYolo, InfYoloResponse
from custom_msg_srv.msg import DetectedObject
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class YoloBrain:
    def __init__(self, model_name='yolov8s-world.pt', sub_topic='', class_names=['bottle']):
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")

        # Initialize YOLOWorld model
        self.model = YOLOWorld(model_name)
        self.model.to(self.device)  # Move model to GPU if available
        self.class_names = class_names
        # Set custom classes
        self.model.set_classes(class_names)
        self.image = None

        # Initialize the CvBridge
        self.bridge = CvBridge()
        # Subscribe to the camera topic
        if sub_topic != '':
            self.image_sub = rospy.Subscriber(sub_topic, Image, self.callback)

    def callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)
        return

    def handle_inference(self, req):
        print("Service call received: Starting YOLO model inference")
        # Run YOLOWorld inference on the image
        results = self.model.predict(self.image, conf=0.25, device=self.device)  # Use the selected device

        # Create a list to hold DetectedObject instances
        detected_objects = []

        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                msg = DetectedObject()
                msg.confidence = box.conf.item()
                cls = box.cls.item() # int that represents the class name
                msg.class_name = self.class_names[int(cls)]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                msg.bbox = [x1, y1, x2, y2]
                detected_objects.append(msg)
        return InfYoloResponse(objects=detected_objects)

    def start_service(self, nodeName='yolo_inference_service', servName='yolo_inference'):
        rospy.init_node(nodeName)
        s = rospy.Service(servName, InfYolo, self.handle_inference)
        rospy.spin()

    def reload_classes(self, classes):
        # Set custom classes eg. ['bottle','glass']
        self.class_names=classes
        self.model.set_classes(self.class_names)

if __name__ == "__main__":
    brain = YoloBrain(sub_topic="/xtion/rgb/image_raw")
    brain.start_service()

