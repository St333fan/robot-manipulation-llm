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
import sys

def call_vit_service():
    try:
        rospy.wait_for_service('/vit_inference')
        vit_service = rospy.ServiceProxy('/vit_inference', InfString)
        
        # Create a request object
        request = InfStringRequest()
        request.question = json.dumps({"prompt":"What happens in the image and what objects can you see? SUPER SHORT ANSWER",
                            "expected_response":{"objects":"object_you_see","environment":"environment_you_see"}})
        request.chat = False
        request.reload = True
        
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
    print("Start... \n\n")
    print(call_vit_service())
    sys.exit()

    is_start = True
    # Define variables for different sections
    modes = {
        "SPEECH": "Only possbile nearby a Human in handshake distance",
        "SCAN": "Scan environment for objects and humans",
        "NAVIGATE": "Move to specified object",
        "PICKUP": "Grab object, location_robot must be at object",
        "PLACE": "Put down held object"
    }

    prompt = "You control a robot by selecting its next operating mode based on the current context. **You try to do the task that is mentioned** Start with the last mode in HISTORY, or START MODE if empty."
    prompt = "You operate a robot by choosing its next mode of operation based on the current context and task requirements. Focus on completing the specified task efficiently. Start from the most recent mode in HISTORY, or default to START MODE if HISTORY is empty."
    instructions = "Reason through what would make the most sense, create a plan if all prerequisites for the whished mode are done.  ANSWER SHORT"#"Reason through what would make the most sense, create a plan if all prerequisites for the whished mode are done.  ANSWER SHORT"
    instructions = "Analyze the situation logically and determine the most suitable next mode. If prerequisites for the desired mode are fulfilled, develop a concise plan of action. Respond succinctly."
    expected_response = {
        "reasoning": "brief_explanation_based_on_the_task",
        "next_mode": "selected_mode",
        "target_object": "if_navigation_needed", #if_navigation_needed
    }

    # Variables for context
    robot_current_task = "None"
    recent_mode = "None"  # Empty history
    detected_objects = [["",""]]  # Empty known objects list
    current_location = "None"  # No specific location
    current_held_object = "None"  # Not holding anything

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
            "history": recent_mode,
            "known_objects": detected_objects,
            "location_robot": current_location,
            "holding_object": current_held_object,
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
            que = json.dumps(question["prompt"])+"\n\n"+json.dumps(question["context"])+"\n\n"+json.dumps(question["expected_response"])
            llm_answer = call_llm_service(question=que, reload=False)
        print(que)
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

                print(f"Reasoning: {reasoning}")
                print(f"Next mode: {next_mode}")
                print(f"Target object: {target_object}")
                #print(f"Rate: {rated}")

            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
        else:
            next_mode = ""
            print("No JSON object found in the response.")

        if next_mode == "SPEECH":
            print("Acquiring a new task...")
            robot_current_task = "Bring the Human a Beer and give it to him"
            # Add logic to acquire a new task
        elif next_mode == "SCAN":
            print("Scanning the environment...")
            detected_objects = [["Human","5m away"], ["Beer","8m away"], ["Kitchen","5m away"]]
            # Add logic to scan the environment
        elif next_mode == "NAVIGATE":
            print("Navigating to the specified object...")
            current_location = target_object  # No specific location
            if target_object =="Human":
                detected_objects = [["Human", "30cm away"], ["Beer", "8m away"], ["Kitchen", "5m away"]]
            elif target_object =="Kitchen":
                detected_objects = [["Human", "5m away"], ["Beer", "8m away"], ["Kitchen", "30cm away"]]
            elif target_object =="Beer":
                detected_objects = [["Human", "5m away"], ["Beer", "30cm away"], ["Kitchen", "5m away"]]
            # Add navigation logic
        elif next_mode == "PICKUP":
            current_held_object = target_object
            print("Picking up the object...")
            # Add logic to pick up the object
        elif next_mode == "PLACE":
            print("Placing the object...")
            break
            # Add logic to place the object
        else:
            print(f"Unknown mode: {next_mode}")
            parse_error = True
            # Handle unexpected mode

        # Variables for context
        #robot_current_task = "Find a Task-Giver"
        recent_modes = next_mode #question["context"]["previous_modes"]+"-->"+str(next_mode)  # Empty history
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
