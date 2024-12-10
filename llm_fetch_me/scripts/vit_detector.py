#!/usr/bin/env python3

import rospy
import ollama
import base64
import time
import cv2
from std_srvs.srv import Trigger, TriggerResponse
from custom_msg_srv.srv import InfString, InfStringResponse
from custom_msg_srv.srv import InfStringImage, InfStringImageResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

'''
        If someone wants to sub to a image topic, use VisionBrain(sub_topic="/xtion/rgb/image_raw")
        If someone wants to send the image with the .srv message, use VisionBrain()
        Adding the InfStringImage still not decided if needed
'''

class VisionBrain:
    def __init__(self, model_name="minicpm-v:8b", system_prompt="", keep_alive='3m', sub_topic=''): #minicpm-v:8b-2.6-fp16
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.keep_alive = keep_alive
        self.messages = []
        self.messages.append({'role': 'system', 'content': self.system_prompt})
        self.image=None
        
        # Initialize the CvBridge
        self.bridge = CvBridge()
        # Subscribe to the camera topic
        if sub_topic!='':
            self.image_sub = rospy.Subscriber(sub_topic, Image, self.callback)
        
    def callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)
        return

    def handle_inference(self, req):
        print("Service call received: Starting ViT model inference")
        if req.reload:
            self.reload()
        if req.chat:
            if self.image==None:
                return InfStringResponse(answer=self.chat(req.question))
            return InfStringResponse(answer=self.chat(req.question, cv_image_to_bytes(self.image))) # should be changed to other
        return InfStringResponse(answer=self.generate_response(req.question, cv_image_to_bytes(self.image)))

    def start_service(self, nodeName='vit_inference_service', servName='vit_inference'):
        rospy.init_node(nodeName)
        s = rospy.Service(servName, InfString, self.handle_inference)
        rospy.spin()

    def generate_response(self, prompt, image, temperature=0.0):
        response = ollama.generate(model=self.model_name, prompt=prompt, system=self.system_prompt, keep_alive=self.keep_alive, images=[image], options={"temperature": temperature})
        generated_text = response['response']
        return generated_text
     
    def chat(self, prompt, image=None):
        if image==None:
            self.messages.append({'role': 'user', 'content': prompt})
        else:
            self.messages.append({'role': 'user', 'content': prompt, 'images': [image]})

        response = ollama.chat(
            model=self.model_name,
            messages=self.messages,
            keep_alive=self.keep_alive,
        )

        generated_text = response['message']['content']
        self.messages.append({'role': 'assistant', 'content': generated_text})
        return generated_text
        
    def reload(self):
        self.messages = []
        self.messages.append({'role': 'system', 'content': self.system_prompt})

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
    brain = VisionBrain(sub_topic="/xtion/rgb/image_raw")
    brain.start_service()

