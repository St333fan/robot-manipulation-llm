#!/usr/bin/env python3

import rospy
import ollama
import base64
import time
import cv2
from std_srvs.srv import Trigger, TriggerResponse
from custom_msg_srv.srv import InfString, InfStringResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VisionBrain:
    def __init__(self, model_name="minicpm-v:8b-2.6-q2_K", system_prompt="", keep_alive='10s'): #minicpm-v:8b-2.6-fp16
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.keep_alive = keep_alive
        self.messages = []
        self.messages.append({'role': 'system', 'content': self.system_prompt})
        self.image = None
        
        # Initialize the CvBridge
        self.bridge = CvBridge()
        
        # Subscribe to the camera topic
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.callback)
        
    def callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)
        return

    def handle_inference(self, req):
        # Print a message to indicate the service call was received
        print("Service call received: Starting AI model inference")
        img = cv_image_to_bytes(self.image)
        return InfStringResponse(answer=self.generate_response(req.question, img))

    def ai_inference_service(self, nodeName='ai_inference_service', servName='vit_inference'):
        rospy.init_node(nodeName)
        s = rospy.Service(servName, InfString, self.handle_inference)
        rospy.spin()

    def generate_response(self, prompt, image):
        response = ollama.generate(model=self.model_name, prompt=prompt, system=self.system_prompt, keep_alive=self.keep_alive, images=[image])
        generated_text = response['response']
        return generated_text
     
    def chat(self, prompt, image): # still in work
        self.messages.append({'role': 'user', 'content': prompt, 'images': [image]})

        response = ollama.chat(
            model=self.model_name,
            messages=self.messages,
            keep_alive=self.keep_alive,
        )

        generated_text = response['message']['content']
        self.messages.append({'role': 'assistant', 'content': generated_text})

        return generated_text

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_to_bytes(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

def cv_image_to_bytes(cv_image, encoding_format='.png'):
    success, buffer = cv2.imencode(encoding_format, cv_image)
    if success:
        return buffer.tobytes()
    else:
        raise ValueError("Image encoding failed")

if __name__ == "__main__":
    brain = VisionBrain()
    brain.ai_inference_service()
