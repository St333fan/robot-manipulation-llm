#!/usr/bin/env python

import rospy
from custom_msg_srv.srv import InfString, InfStringRequest
from custom_msg_srv.srv import InfYolo, InfYoloRequest
from custom_msg_srv.msg import DetectedObject
from geometry_msgs.msg import Twist
from numpy.lib.type_check import isreal
from sensor_msgs.msg import Image
import time

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

def call_llm_service(information="", reload=False, question=''):
    try:
        rospy.wait_for_service('/llm_inference')
        llm_service = rospy.ServiceProxy('/llm_inference', InfString)
        
        # Create a request object
        request = InfStringRequest()
        request.question = question
        request.chat = True
        request.reload = reload
        
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
       
# Function to publish a message to the /mobile_base_controller/cmd_vel topic
def publish_cmd_vel(linear_x=0.0, linear_y=0.0, linear_z=0.0, angular_x=0.0, angular_y=0.0, angular_z=0.0, duration=1.0):
    # Create a publisher object
    pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
    rospy.sleep(1)  # Wait for the publisher to properly connect

    # Create and configure the Twist message
    cmd = Twist()
    cmd.linear.x = linear_x
    cmd.linear.y = linear_y
    cmd.linear.z = linear_z
    cmd.angular.x = angular_x
    cmd.angular.y = angular_y
    cmd.angular.z = angular_z

    rospy.loginfo("Publishing to /mobile_base_controller/cmd_vel for %s seconds", duration)

    # Publish the message for the specified duration
    start_time = time.time()
    while time.time() - start_time < duration and not rospy.is_shutdown():
        pub.publish(cmd)
        rospy.sleep(0.1)  # Publish at 10 Hz

    # Stop the robot after the duration
    stop_cmd = Twist()
    pub.publish(stop_cmd)
    rospy.loginfo("Stopped publishing")


def main(): # add a Hoistory of some past taken actions and add a time to them
    is_start = True
    question = "You can call--> call_vit_service(): for analysing camera frame, call_yolo_service(): for finding object, publish_cmd_vel(): for driving around. Past retreaved information from function: "
    question = "You can call-->  call_yolo_service(): for finding object, publish_cmd_vel(): for driving around. SUPER SHORT ANSWER ONLY WRITE WHISHED FUNCITON Past retreaved information from function: "
    goal = 'Your Goal is to find a red can!'
    history = ''


    rospy.init_node('prefrontal_cortex_node', anonymous=True)
    # Initialize the ROS node
    rospy.loginfo("Node started and will call services in a loop.")

    rate = rospy.Rate(1)  # Adjust the rate as needed (e.g., 1 Hz)
    information = ""
    while not rospy.is_shutdown():
        if is_start:
            llm_answer = call_llm_service(question=(goal+"\n\n"+question+"None"), reload=True)
            is_start = False
        else:
            llm_answer = call_llm_service(question=question+str(information))
        print(llm_answer)
        
        if "call_vit_service" in llm_answer:
            information = call_vit_service()
            print(information)
        elif "call_yolo_service" in llm_answer:
            information = call_yolo_service()
            print(information)
        elif "publish_cmd_vel" in llm_answer:
            publish_cmd_vel(linear_x=0.5, angular_z=0.2)
            time.sleep(2)
            print('drove')
            information = 'robot drove'

        rospy.loginfo("All services have been called in this cycle.")
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
