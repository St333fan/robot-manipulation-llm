#!/usr/bin/env python

import rospy
from custom_msg_srv.srv import InfString, InfStringRequest
from custom_msg_srv.srv import InfYolo, InfYoloRequest
from custom_msg_srv.msg import DetectedObject
from geometry_msgs.msg import Twist
from numpy.lib.type_check import isreal
from sensor_msgs.msg import Image
import time
import json
import re

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

## I tried unstructured but it is trash
def main(): # add a Hoistory of some past taken actions and add a time to them
    is_start = True
    # Define variables for different sections
    modes = {
        "ASK_FOR_TASK": "Get new task (must be near relevant object/location)",
        "SCAN": "Scan current environment for objects",
        "NAVIGATE": "Move to specified object",
        "PICKUP": "Grab object (must be nearby)",
        "PLACE": "Put down held object",
        "GET_HISTORY": "Review past actions"
    }

    prompt = "You control a robot by selecting its next operating mode based on the current context. Start with the last mode in HISTORY, or START MODE if empty."

    instructions = "Reason through what would make the most sense, create a plan if all prerequisites for the whished mode are done.  ANSWER SHORT"

    expected_response = {
        "reasoning": "brief_explanation_based_on_the_given_information",
        "next_mode": "selected_mode",
        "target_object": "if_navigation_needed",
        "judge": "brief_rating_of_1-10"
    }

    # Variables for context
    robot_current_task = "Scan/find/drive to Human and get a Task"
    recent_actions = ["None"]  # Empty history
    detected_objects = ["None"]  # Empty known objects list
    current_location = None  # No specific location
    current_held_object = None  # Not holding anything

    parse_error = False

    rospy.init_node('prefrontal_cortex_node', anonymous=True)
    # Initialize the ROS node
    rospy.loginfo("Node started and will call services in a loop.")

    rate = rospy.Rate(1)  # Adjust the rate as needed (e.g., 1 Hz)
    information = ""
    while not rospy.is_shutdown():
        context = {
            "modes": modes,
            "task": robot_current_task,
            "history": recent_actions,
            "known_objects": detected_objects,
            "location_robot": current_location,
            "holding_object": current_held_object
        }

        # Construct the full JSON object
        question = {
            "prompt": prompt,
            "context": context,
            "instructions": instructions,
            "expected_response": expected_response
        }

        if parse_error:
            call_llm_service("ERROR: could not parse json, the \"next_mode\" and \"reasoning\", answer again!", reload=False)
            parse_error = False
        else:
            llm_answer = call_llm_service(question=json.dumps(question), reload=False)
        print(json.dumps(question))
        print(llm_answer)

        # Extract JSON using regex
        json_match = re.search(r'\{.*\}', llm_answer, re.DOTALL)

        if json_match:
            json_string = json_match.group()
            try:
                # Parse the JSON
                response = json.loads(json_string)

                # Access the elements
                next_mode = response["next_mode"]
                target_object = response["target_object"]
                reasoning = response["reasoning"]
                rated = response["judge"]

                print(f"Reasoning: {reasoning}")
                print(f"Next mode: {next_mode}")
                print(f"Target object: {target_object}")
                print(f"Rate: {rated}")

            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
        else:
            next_mode = ""
            print("No JSON object found in the response.")

        if next_mode == "ASK_FOR_TASK":
            print("Acquiring a new task...")
            robot_current_task = "Bring the Human a Beer"
            # Add logic to acquire a new task
        elif next_mode == "SCAN":
            print("Scanning the environment...")
            detected_objects = ["Kitchen", "Human", "Beer"]
            # Add logic to scan the environment
        elif next_mode == "NAVIGATE":
            print("Navigating to the specified object...")
            current_location = target_object  # No specific location
            # Add navigation logic
        elif next_mode == "PICKUP":
            current_held_object = target_object
            print("Picking up the object...")
            # Add logic to pick up the object
        elif next_mode == "PLACE":
            print("Placing the object...")
            break
            # Add logic to place the object
        elif next_mode == "GET_HISTORY":
            print("Retrieving history of actions...")
            # Add logic to get history
        else:
            print(f"Unknown mode: {next_mode}")
            parse_error = True
            # Handle unexpected mode

        # Variables for context
        #robot_current_task = "Find a Task-Giver"
        recent_actions = [next_mode]  # Empty history
        #detected_objects = []  # Empty known objects list
        #current_location = None  # No specific location
        #current_held_object = None  # Not holding anything

        rospy.loginfo("All services have been called in this cycle.")
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
