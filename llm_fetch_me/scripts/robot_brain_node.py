#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.expanduser('~/exchange/lmt_ws/src/llm_fetch_me/scripts'))

import rospy
from custom_msg_srv.srv import InfString, InfStringRequest
from custom_msg_srv.srv import InfYolo, InfYoloRequest
from custom_msg_srv.msg import DetectedObject
from sensor_msgs.msg import Image

def call_vit_service():
    try:
        rospy.wait_for_service('/vit_inference')
        vit_service = rospy.ServiceProxy('/vit_inference', InfString)
        
        # Create a request object
        request = InfStringRequest()
        request.question = "what do you see"
        request.chat = False
        request.reload = False
        
        response = vit_service(request)
        rospy.loginfo("Vit called successfully.")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call Vit service: %s" % e)
    return response.answer

def call_llm_service():
    try:
        rospy.wait_for_service('/llm_inference')
        llm_service = rospy.ServiceProxy('/llm_inference', InfString)
        
        # Create a request object
        request = InfStringRequest()
        request.question = "What are you?"
        request.chat = False
        request.reload = False
        
        response = llm_service(request)
        rospy.loginfo("LLM called with response:")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call LLM service: %s" % e)
    return response.answer

def call_yolo_service():
    try:
        rospy.wait_for_service('/yolo_inference')
        trigger_service = rospy.ServiceProxy('/yolo_inference', InfYolo)
        
        # Create a request object
        request = InfYoloRequest()
        img = Image()
        request.image = img
        
        response = trigger_service(InfYoloRequest())
        rospy.loginfo("Yolo called with response:")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call Yolo service: %s" % e)
        
     # Check if the 'objects' list is not empty
    if response.objects:
        return response.objects[0].class_name + " -- " + str(response.objects[0].depth)
    else:
        print("No objects detected in the response")
        return None

def main():
    rospy.init_node('prefrontal_cortex_node', anonymous=True)

    rospy.loginfo("Node started and will call services in a loop.")

    rate = rospy.Rate(1)  # Adjust the rate as needed (e.g., 1 Hz)
    while not rospy.is_shutdown():
        #print(call_vit_service())
        #print(call_llm_service())
        print(call_yolo_service())
        rospy.loginfo("All services have been called in this cycle.")
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
