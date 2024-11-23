#!/usr/bin/env python

import rospy
from custom_msg_srv.srv import InfString, InfStringRequest
from custom_msg_srv.srv import InfYolo, InfYoloRequest
from custom_msg_srv.msg import DetectedObject
from fontTools.ttLib.tables.ttProgram import instructions
from geometry_msgs.msg import Twist
from numpy.lib.type_check import isreal
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler
import math
import time
import json
import yaml
import numpy as np
import re
import sys
import ast
from sympy.codegen.ast import continue_


def call_vit_service(question="What happens in the image and what objects can you see? SUPER SHORT ANSWER"):
    try:
        rospy.wait_for_service('/vit_inference')
        vit_service = rospy.ServiceProxy('/vit_inference', InfString)
        
        # Create a request object
        request = InfStringRequest()
        request.question = question
        #json.dumps({"prompt":"What happens in the image and what objects can you see? SUPER SHORT ANSWER",
                            #"expected_response":{"objects":"object_you_see","environment":"environment_you_see"}})
        request.chat = False
        request.reload = True
        
        response = vit_service(request)
        rospy.loginfo("Vit called successfully.")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call Vit service: %s" % e)
    return response.answer

def call_llm_service(information="", reload=False, question='', chat=True):
    try:
        rospy.wait_for_service('/llm_inference')
        llm_service = rospy.ServiceProxy('/llm_inference', InfString)
        
        # Create a request object
        request = InfStringRequest()
        request.question = question
        request.chat = chat
        request.reload = reload
        
        response = llm_service(request)
        rospy.loginfo("LLM called with response:")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call LLM service: %s" % e)
    return response.answer

def call_yolo_service(classes=["bottle"]):
    try:
        rospy.wait_for_service('/yolo_inference')
        trigger_service = rospy.ServiceProxy('/yolo_inference', InfYolo)
        print(classes)

        # Create a request object
        request = InfYoloRequest()
        img = Image()
        request.image = img
        request.classes = classes
        request.confidence = 0.1
        
        response = trigger_service(request)
        rospy.loginfo("Yolo called with response:")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call Yolo service: %s" % e)
        return None

        # Initialize an empty list to store the objects
    obj = []

    # Check if the 'objects' list is not empty
    if response.objects:
        for detected_object in response.objects:
            # Append each object's class_name and depth as a sublist
            obj.append([detected_object.class_name, detected_object.depth])
    else:
        print("No objects detected in the response")
    return obj
       
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

def scan_environment(task_objects=["bottle"]):
    print("VIT")
    vit_results = call_vit_service()

    prompt = "Extract all key objects(humans are objects too!) and environments from the Vision Transformer's scene description. Focus on common objects and spaces in a room."
    context = {"vit_results":vit_results}
    instructions = "Reason through what would make the most sense which objects/spaces are needed. Convert synonyms to common terms (e.g., 'people' to 'human') ANSWER SHORT"
    expected_response={
        "reasoning": "brief_explanation",
        "detect_objects": "[\"relevant_objects\"]",
        "detect_space": "[\"relevant_higher_level_spaces(e.g.kitchen/bedroom/etc\"]"
    }

    question = {
        "prompt": prompt,
        "context": context,
        "instructions": instructions,
        "expected_response": expected_response
    }
    print(vit_results)
    print(expected_response) #+ "\n\n" + json.dumps(question["instructions"])
    que = json.dumps(question)
    llm_answer = call_llm_service(question=que, chat=False)
    #llm_answer = call_llm_service(question="Responde with the json format", reload=False)
    print(llm_answer)
    # Extract JSON using regex
    json_match = re.search(r'\{.*\}', llm_answer, re.DOTALL)

    if json_match:
        json_string = json_match.group()
        try:
            # Parse the JSON
            response = json.loads(json_string)

            # Access the elements
            detect_objects = response["detect_objects"]+task_objects
            #detect_objects.append(response["detect_objects"])
            detect_space = response["detect_space"]
            reasoning = response["reasoning"]

            print(f"Reasoning: {reasoning}")
            print(f"Obj: {detect_objects}")
            print(f"Spaces: {detect_space}")
            # print(f"Rate: {rated}")

        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
    else:
        print("No JSON object found in the response.")
        return call_yolo_service(task_objects)

    return call_yolo_service(detect_objects)
    #return call_yolo_service(["bottle","table","white object"])

def aquire_task():
    task = ""
    human_response = "None"
    vit_results = call_vit_service(question="Do the humans look towards the camera?")

    prompt = "Determine if human is actively ready to interact based on visual cues. Use the results from a ViT."
    context = {"vit_results": vit_results,
               "human_response":human_response}

    instructions = "Reason through what would make the most sense, decide if you should start speaking with them, if yes, ask them for a task. The human needs to look at you activly! ANSWER SHORT"
    expected_response = {
        "reasoning": "brief_explanation",
        "speech": "only_yes_or_no",
        "get_attention": "only_yes_or_no",
        "question": "potential_question"
    }

    question = {
        "prompt": prompt,
        "context": context,
        "instructions": instructions,
        "expected_response": expected_response
    }
    print(vit_results)
    print(expected_response)  # + "\n\n" + json.dumps(question["instructions"])
    que = json.dumps(question)
    llm_answer = call_llm_service(question=que, chat=False)

    # llm_answer = call_llm_service(question="Responde with the json format", reload=False)
    print(llm_answer)
    # Extract JSON using regex
    json_match = re.search(r'\{.*\}', llm_answer, re.DOTALL)

    if json_match:
        json_string = json_match.group()
        try:
            # Parse the JSON
            response = json.loads(json_string)

            # Access the elements
            get_attention = response["get_attention"]
            question = response["question"]
            get_speech = response["speech"]
            reasoning = response["reasoning"]

            print(f"get_attention: {get_attention}")
            print(f"question: {question}")
            print(f"get_speech: {get_speech}")
            print(f"reasoning: {reasoning}")
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return "Error decoding JSON:"
    else:
        print("No JSON object found in the response.")
        return "No JSON object found in the response."

    if get_attention == "yes": # need to add speech input
        task = "Bring the Human a Bottle"

    return task

def send_rotation_goal(goal_pub, angle_degrees):
    """
    Sends a rotation goal using current position from /robot_pose and desired angle.

    Args:
        angle_degrees (float): Desired rotation angle in degrees
    """
    try:
        # Get current robot position
        current_pose = rospy.wait_for_message("/robot_pose", PoseWithCovarianceStamped, timeout=1.0)

        # Create goal with current position
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        # Use current x, y, z position
        goal.pose.position.x = current_pose.pose.pose.position.x
        goal.pose.position.y = current_pose.pose.pose.position.y
        goal.pose.position.z = current_pose.pose.pose.position.z

        # Set new orientation for rotation
        angle_rad = math.radians(angle_degrees)
        quaternion = quaternion_from_euler(0, 0, angle_rad)
        print(quaternion)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        #goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        rospy.sleep(2)  # Ensure connection is established
        # Publish goal
        goal_pub.publish(goal)


        rospy.loginfo(f"Rotating to {angle_degrees} degrees at current position: "
                     f"x={goal.pose.position.x:.2f}, y={goal.pose.position.y:.2f}")

    except rospy.ROSException as e:
        rospy.logwarn(f"Failed to get current robot pose: {str(e)}")

def object_distance():
    try:
        # Get current robot position
        current_pose = rospy.wait_for_message("/robot_pose", PoseWithCovarianceStamped, timeout=1.0)

    except rospy.ROSException as e:
        rospy.logwarn(f"Failed to get current robot pose: {str(e)}")

    return None

def point_to_edge_distance_with_buffer(point, edge_start, edge_end, buffer_distance):
    """
    Compute the closest point on an edge to a given point and adjust it with a buffer.
    """
    edge = edge_end - edge_start
    edge_length_squared = np.dot(edge, edge)
    if edge_length_squared == 0:  # Edge start and end are the same point
        closest_point = edge_start
    else:
        # Project point onto the line (parameterized by t)
        t = np.dot(point - edge_start, edge) / edge_length_squared
        t = np.clip(t, 0, 1)  # Clamp t to stay within the edge segment

        # Find the closest point on the edge
        closest_point = edge_start + t * edge

    # Adjust the closest point outward by buffer_distance
    edge_normal = np.array([-edge[1], edge[0]])  # Perpendicular vector
    edge_normal = edge_normal / np.linalg.norm(edge_normal)  # Normalize

    buffer_adjusted_point = closest_point + buffer_distance * edge_normal
    distance = np.linalg.norm(point - buffer_adjusted_point)
    return buffer_adjusted_point, distance

def find_nearest_outside_point_with_buffer(polygon, point_inside, buffer_distance):
    """
    Find the nearest point outside the polygon to the given point inside, with a buffer.
    """
    closest_point = None
    min_distance = float('inf')

    # Iterate over the polygon's edges
    for i in range(len(polygon)):
        edge_start = polygon[i]
        edge_end = polygon[(i + 1) % len(polygon)]  # Wrap around to form edges
        candidate_point, distance = point_to_edge_distance_with_buffer(point_inside, edge_start, edge_end,
                                                                       buffer_distance)
        if distance < min_distance:
            min_distance = distance
            closest_point = candidate_point

    return closest_point

def parse_table_coordinates(yaml_file):
    """
    Parse table coordinates from YAML file.
    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    table1_coords = []
    table2_coords = []

    submap = data['vo']['submap_0']

    for key, value in submap.items():
        coords = value[2:4]
        if 'table_1' in key:
            table1_coords.append(coords)
        elif 'table_2' in key:
            table2_coords.append(coords)

    return np.array(table1_coords), np.array(table2_coords)

## I tried unstructured but it is trash
def main(): # add a History of some past taken actions and add a time to them
    rospy.init_node('prefrontal_cortex_node', anonymous=True)
    # Initialize the ROS node
    rospy.loginfo("Node started and will call services in a loop.")

    print("Start... \n\n")

    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    send_rotation_goal(goal_pub=goal_pub, angle_degrees=180)

    sys.exit()
    print(aquire_task())
    #print(scan_environment())
    #print(call_vit_service())


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
