#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO, YOLOWorld
import torch
import cv2

class BeerCanDetector:
    def __init__(self):
        rospy.init_node('beer_can_detector', anonymous=True)
        
        # Initialize the YOLO model
        self.model = YOLO('yolov8n.pt')  # You can change this to a custom-trained model for beer cans
        
        # Initialize the CvBridge
        self.bridge = CvBridge()
        
        # Subscribe to the camera topic
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.callback)
        
        # Publisher for the processed image
        self.image_pub = rospy.Publisher("/beer_can_detections", Image, queue_size=10)

    def callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as
            rospy.logerr(e)
            return

        # Run YOLOv8 inference on the image
        results = self.model(cv_image)

        # Process the results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                conf = box.conf[0]
                xyxy = box.xyxy[0]

                # Check if the detected object is a beer can (assuming class ID 39 is 'bottle')
                # You may need to adjust this based on your model's class IDs
                if class_id == 39 and conf > 0.5:  # You can adjust the confidence threshold
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, f'Beer Can: {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the processed image back to ROS Image message and publish
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except Exception as e:
       	    print('err')
            rospy.logerr(e)

    def run(self):
        rospy.spin()
        
class RedCanDetector:
    def __init__(self):
        rospy.init_node('can_detector', anonymous=True)
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")
        
        # Initialize YOLOWorld model
        self.model = YOLOWorld('yolov8x-world.pt')
        self.model.to(self.device)  # Move model to GPU if available
        
        # Set custom classes
        self.model.set_classes(["bottle"])
        
        # Initialize the CvBridge
        self.bridge = CvBridge()
        
        # Subscribe to the camera topic
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.callback)
        
        # Publisher for the processed image
        self.image_pub = rospy.Publisher("/red_can_detections", Image, queue_size=10)

    def callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        # Run YOLOWorld inference on the image
        results = self.model.predict(cv_image, conf=0.25, device=self.device)  # Use the selected device

        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf.item()
                cls = box.cls.item()
                if conf > 0.1:  # You can adjust this threshold
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle
                    cv2.putText(cv_image, f'Can: {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert the processed image back to ROS Image message and publish
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except Exception as e:
            rospy.logerr(e)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    detector = RedCanDetector()
    detector.run()
