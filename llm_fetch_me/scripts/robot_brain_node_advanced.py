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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from matplotlib.path import Path

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

def call_yolo_service(classes=["bottle"], with_xy = False):
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
            if with_xy:
                obj.append([detected_object.class_name, detected_object.depth, detected_object.x, detected_object.y])
            else:
                obj.append([detected_object.class_name, detected_object.depth])
    else:
        print("No objects detected in the response")
    return obj

def call_whisper():
    return None

# Function to publish a message to the /mobile_base_controller/cmd_vel topic # for now it is a test function
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

def scan_environment(task_objects=["bottle"]): # add objects taken from task or give it the Task Reasonong objects
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

    return (detect_objects)
    #return call_yolo_service(["bottle","table","white object"])

def scan_environment_pipeline():
    """
    :return: all found objects in environment
    """
    obj = []
    # - for right + for left
    send_rotation_goal(30)
    rospy.sleep(10)
    obj.append(scan_environment())

    send_rotation_goal(-90)
    rospy.sleep(10)
    obj.append(scan_environment())

    print(obj)
    return obj

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

def send_rotation_goal(angle_degrees):
    """
    Sends an incremental rotation goal using current position from /robot_pose and desired angle.

    Args:
        goal_pub (Publisher): ROS Publisher to publish the goal.
        angle_degrees (float): Desired incremental rotation angle in degrees.
    """
    try:
        goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        # Get current robot pose
        current_pose = rospy.wait_for_message("/robot_pose", PoseWithCovarianceStamped, timeout=1.0)

        # Extract current orientation quaternion
        current_orientation = current_pose.pose.pose.orientation
        current_quaternion = [
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        ]

        # Convert current quaternion to Euler angles
        _, _, current_yaw = euler_from_quaternion(current_quaternion)

        # Calculate new yaw as the sum of current yaw and desired incremental angle
        angle_rad = math.radians(angle_degrees)
        new_yaw = current_yaw + angle_rad

        # Convert new yaw to quaternion
        new_quaternion = quaternion_from_euler(0, 0, new_yaw)

        # Create and populate the goal message
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        # Maintain current position
        goal.pose.position.x = current_pose.pose.pose.position.x
        goal.pose.position.y = current_pose.pose.pose.position.y
        goal.pose.position.z = current_pose.pose.pose.position.z

        # Set new orientation for rotation
        goal.pose.orientation.x = new_quaternion[0]
        goal.pose.orientation.y = new_quaternion[1]
        goal.pose.orientation.z = new_quaternion[2]
        goal.pose.orientation.w = new_quaternion[3]

        rospy.sleep(2)  # Ensure connection is established

        # Publish goal
        goal_pub.publish(goal)

        rospy.loginfo(f"Incremental rotation of {angle_degrees} degrees applied. New yaw: {math.degrees(new_yaw):.2f}")

    except rospy.ROSException as e:
        rospy.logwarn(f"Failed to get current robot pose: {str(e)}")

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

def parse_table_coordinates(yaml_path):
    """
    Parse table coordinates from YAML file.
    """
    with open(yaml_path, 'r') as file:
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

def update_location(): # I only need when I do class
    return None

def update_held_object(): # I only need when I do class
    return None

def find_object_pose(objects=[["", 0.0]]):
    '''
    :param objects: objects with name and distance(depth)
    :return: dictionary of objects and its corresponding pose PoseStamped()
    '''
    try:
        # Get current robot position
        current_pose = rospy.wait_for_message("/robot_pose", PoseWithCovarianceStamped, timeout=1.0)

        # Extract robot's current x, y position and orientation (yaw)
        robot_x = current_pose.pose.pose.position.x
        robot_y = current_pose.pose.pose.position.y

        # Get the orientation quaternion and convert it to yaw
        orientation = current_pose.pose.pose.orientation
        yaw = math.atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                         1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2))

        objects_pose = {}

        for obj_name, distance in objects:
            # Calculate object's global position
            object_x = robot_x + distance * math.cos(yaw)
            object_y = robot_y + distance * math.sin(yaw)

            # Create a PoseStamped message
            obj_pose = PoseStamped()
            obj_pose.header.frame_id = "map"  # Assuming the map frame for global coordinates
            obj_pose.pose.position.x = object_x
            obj_pose.pose.position.y = object_y
            obj_pose.pose.position.z = 0.0  # Assuming 2D plane

            # Set orientation to face the same direction as the robot
            obj_pose.pose.orientation = orientation

            objects_pose[obj_name] = obj_pose

    except rospy.ROSException as e:
        rospy.logwarn(f"Failed to get current robot pose: {str(e)}")
        return {}
    return objects_pose

def objects_pose_to_distance(all_objects={"No_Object": PoseStamped()}):
    """
    Turns the Pose into distance, for LLM prompting
    """
    try:
        # Get current robot position
        current_pose_msg = rospy.wait_for_message("/robot_pose", PoseWithCovarianceStamped, timeout=1.0)
        current_pose = current_pose_msg.pose.pose

        objects_distance = []

        for object_name, object_pose in all_objects.items():
            # Calculate Euclidean distance
            dx = current_pose.position.x - object_pose.pose.position.x
            dy = current_pose.position.y - object_pose.pose.position.y
            dz = current_pose.position.z - object_pose.pose.position.z
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # Append result as [Object_Name, distance]
            objects_distance.append([object_name, distance])

        return objects_distance

    except rospy.ROSException as e:
        rospy.logwarn(f"Failed to get current robot pose: {str(e)}")
        return []

def drive_to(location=PoseStamped()):
    """
    Publishes a PoseStamped message to the /move_base_simple/goal topic.

    Args:
        location (PoseStamped): Target location in the form of a PoseStamped message.

    Returns:
        None
    """
    try:
        # Initialize publisher
        pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # Ensure the publisher connection is established
        rospy.sleep(1.0)  # Short delay to allow for connection

        # Publish the target location
        pub.publish(location)
        rospy.loginfo(f"Published target location to /move_base_simple/goal: {location}")

    except rospy.ROSException as e:
        rospy.logwarn(f"Failed to publish target location: {str(e)}")

    return None

def navigate_to_point(target_pose=PoseStamped(), buffer_distance=1):
    """
    Determines the navigation point based on the target's relation to tables.
    target_pose: PoseStamped, target point in map frame
    buffer_distance: float, buffer distance to keep from tables or shorten drive

    When the point is in a forbidden zone it drives to another point outside, when it drives to a location
    which is not it will stop buffer_distance away
    """
    # Extract position from PoseStamped
    target_point = np.array([target_pose.pose.position.x, target_pose.pose.position.y])

    # Parse table coordinates
    table1_coords, table2_coords = parse_table_coordinates('/home/user/exchange/map_1/mmap.yaml')

    try:
        # Get current robot position
        current_pose_msg = rospy.wait_for_message("/robot_pose", PoseWithCovarianceStamped, timeout=1.0)
        current_pose = current_pose_msg.pose.pose
        robot_position = np.array([current_pose.position.x, current_pose.position.y])
    except rospy.ROSException as e:
        rospy.logwarn(f"Failed to get current robot pose: {str(e)}")
        return  # Exit if robot position is unavailable

    if Path(table1_coords).contains_point(target_point):
        rospy.loginfo("Point is inside Table 1, navigating to nearest outside point.")
        point_to_drive = find_nearest_outside_point_with_buffer(table1_coords, target_point, buffer_distance)
    elif Path(table2_coords).contains_point(target_point):
        rospy.loginfo("Point is inside Table 2, navigating to nearest outside point.")
        point_to_drive = find_nearest_outside_point_with_buffer(table2_coords, target_point, buffer_distance)
    else:
        rospy.loginfo("Point is outside both tables, navigating to a closer point.")
        # Compute vector from robot to target
        vector_to_target = target_point - robot_position
        distance_to_target = np.linalg.norm(vector_to_target)

        if distance_to_target > buffer_distance:
            # Shorten the distance by buffer_distance
            point_to_drive = robot_position + (vector_to_target / distance_to_target) * (
                        distance_to_target - buffer_distance)
        else:
            rospy.logwarn("Target is within buffer distance; navigating directly to target.")
            point_to_drive = target_point

    # Convert point to PoseStamped
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = point_to_drive[0]
    pose.pose.position.y = point_to_drive[1]
    pose.pose.position.z = 0.0
    pose.pose.orientation.w = 1.0  # Default forward orientation

    drive_to(pose)

def publish_head_position():
    """
    def switch_controllers(start_controllers, stop_controllers):
    try:
        switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

        # Switch controllers
        response = switch_controller(
            start_controllers=start_controllers,
            stop_controllers=stop_controllers,
            strictness=0  # 0 means allow partial switching
        )

        print(f"Controller switch response: {response}")
        return response
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return False

def publish_head_position():
    # Create publisher
    pub_head = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=10)

    # Create JointTrajectory message
    head_cmd = JointTrajectory()
    head_cmd.header.stamp = rospy.Time.now()
    head_cmd.joint_names = ['head_1_joint', 'head_2_joint']

    # Create trajectory point
    point = JointTrajectoryPoint()
    point.positions = [0.0, -0.3]  # Specific head joint positions
    point.time_from_start = rospy.Duration(1, 0)

    head_cmd.points.append(point)

    # Publish multiple times to ensure receipt
    for _ in range(3):
        pub_head.publish(head_cmd)
        print("Published head position")
        rospy.sleep(0.5)
    :return:
    """
    # Create publisher for head controller
    pub_head = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=10)

    # Create JointTrajectory message
    head_cmd = JointTrajectory()
    head_cmd.header.seq = 0
    head_cmd.header.stamp = rospy.Time.now()
    head_cmd.joint_names = ['head_1_joint', 'head_2_joint']

    # Create trajectory point
    point = JointTrajectoryPoint()
    point.positions = [0.0, -0.3]  # Same as the rostopic pub command
    point.time_from_start = rospy.Duration(1, 0)

    head_cmd.points.append(point)

    # Publish head position
    pub_head.publish(head_cmd)
    print("Published head position")

def fine_positioning(obj_search=["Coca Cola can"]):
    # Create publisher
    pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)

    print("Starting fine positioning")
    publish_head_position()
    # Create Twist commands
    rotate_cmd = Twist()
    rotate_cmd.angular.z = 0.3  # Rotate at 0.1 rad/s
    stop_cmd = Twist()  # Stop command (all velocities set to 0)

    rate = rospy.Rate(3)  # 3 Hz control loop

    # Publish rotation command
    pub.publish(rotate_cmd)
    rate.sleep()

    while not rospy.is_shutdown():
        print("Searching for object")
        object_list = call_yolo_service(obj_search)  # Get detected objects and distances
        print(object_list)
        # Check if the target object is in the detected list and within 1 meter
        for obj in object_list:
            obj_name, distance = obj
            if obj_name in obj_search and distance < 1.4:
                pub.publish(stop_cmd)  # Send stop command

                while not rospy.is_shutdown(): # bad coding
                    object_list = call_yolo_service(obj_search, with_xy=True)
                    print(object_list)

                    # Check if the target object is in the detected list and within 1 meter
                    for obj in object_list:
                        obj_name, distance, x, y = obj
                        if obj_name in obj_search and distance < 1.4:
                            if x < 230:
                                rotate_cmd.angular.z = 0.1
                            elif x > 410:
                                rotate_cmd.angular.z = -0.1
                            else:
                                pub.publish(stop_cmd)  # Send stop command
                                return None  # Exit once the object is found within range
                            rate = rospy.Rate(3)  # 3 Hz control loop
                            # Publish rotation command
                            pub.publish(rotate_cmd)
                            rate.sleep()

        # Continue rotating if object not found
        pub.publish(rotate_cmd)
        rate.sleep()

    return None

def speech_input():
    str = ""
    str = "Bring the Human a Beer and give it to him"
    return str


""" NON DYNAMIC ENVIRONMENT
    Start
    Scan() N-times
        send_rotation_goal()
        scan
        Turn Right Table and Bottle 
        Save the YOLO Data with the Depth
        send_rotation_goal()
        scan
        Turn Left Table and Human?
        Save the YOLO Data with the Depth
    Calculate the Position of the Objects in 3D Mapping Space update_location_object, no needed camera in-and extrinsics
    Drive to Human (Calculate a point to drive to) and (update robo loc)
    aquire_task()
        give just text back, no ViT looking and no SpeechInput will be tested in Reallife or I could place Images
    update the task
    Drive to Bottle (Here the nearest_outside_point_with_buffer test)
    positioning() todo function for finding the best angle, can be done with YOLO should run reallife
    Just test fake gripping or test real one, positioning is Key
    Drive to Human with Bottle
    Drop it!
"""


## I tried unstructured but it is trash
def test(): # add a History of some past taken actions and add a time to them
    rospy.init_node('prefrontal_cortex_node', anonymous=True)
    # Initialize the ROS node
    rospy.loginfo("Node started and will call services in a loop.")

    print("Start... \n\n")
    fine_positioning()
    #navigate_to_point()
    #scan_environment_pipeline()

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
        recent_mode = next_mode #question["context"]["previous_modes"]+"-->"+str(next_mode)  # Empty history
        #detected_objects = []  # Empty known objects list
        #current_location = None  # No specific location
        #current_held_object = None  # Not holding anything

        rospy.loginfo("All services have been called in this cycle.")
        rate.sleep()

def main(): # add a History of some past taken actions and add a time to them
    rospy.init_node('prefrontal_cortex_node', anonymous=True)
    # Initialize the ROS node
    rospy.loginfo("Node started and will call services in a loop.")

    print("Start... \n\n")
    fine_positioning()
    #navigate_to_point()
    #scan_environment_pipeline()

    sys.exit()
    print(aquire_task())
    #print(scan_environment())
    #print(call_vit_service())

    all_found_objects_with_twist = {}

    is_start = True
    # Define variables for different sections
    modes = {
        "SPEECH": "Only possible nearby a Human in handshake distance",
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
            robot_current_task = speech_input()
            # Add logic to acquire a new task
        elif next_mode == "SCAN":
            print("Scanning the environment...")
            detected_objects = scan_environment_pipeline() # [["Human","5"], ["Beer","8"], ["Kitchen","5"]...]

            # Transform the data to match the required style
            detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]
            #detected_objects = [["Human","5m away"], ["Beer","8m away"], ["Kitchen","5m away"]]

            if len(all_found_objects_with_twist) == 0:
                all_found_objects_with_twist = find_object_pose(detected_objects)
            else:
                all_found_objects_with_twist.update(find_object_pose(detected_objects))

            # Add logic to scan the environment
        elif next_mode == "NAVIGATE": # make a difference from space to object, cant drive to near to space but can
            # drive near to object
            print("Navigating to the specified object...")
            navigate_to_point(traget_pose=all_found_objects_with_twist[target_object])
            if traget_object != "kitchen":
                fine_positioning(obj_search=[target_object])
            current_location = target_object  # No specific location
            detected_objects = objects_pose_to_distance(all_found_objects_with_twist)

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
        recent_mode = next_mode #question["context"]["previous_modes"]+"-->"+str(next_mode)  # Empty history
        #detected_objects = []  # Empty known objects list
        #current_location = None  # No specific location
        #current_held_object = None  # Not holding anything

        rospy.loginfo("All services have been called in this cycle.")
        rate.sleep()

if __name__ == '__main__':
    try:
        test()
        # main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
