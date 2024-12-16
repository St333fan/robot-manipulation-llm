#!/usr/bin/env python

import rospy
from custom_msg_srv.srv import InfString, InfStringRequest
from custom_msg_srv.srv import InfYolo, InfYoloRequest
from custom_msg_srv.msg import DetectedObject
from fontTools.ttLib.tables.ttProgram import instructions
from geometry_msgs.msg import Twist
from numpy.lib.type_check import isreal
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, Point, Quaternion, PoseWithCovarianceStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from matplotlib.path import Path
from std_msgs.msg import Header
from std_srvs.srv import Empty

from controller_manager_msgs.srv import SwitchController, UnloadController

import os
import math
import subprocess
import time
import json
import yaml
import numpy as np
import re
import sys
import ast
import tf2_ros
import signal
from sympy.codegen.ast import continue_

# for speech
import actionlib
from pal_interaction_msgs.msg import TtsAction, TtsActionGoal
from actionlib_msgs.msg import GoalID


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
        request.confidence = 0.5
        
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
                obj.append([detected_object.class_name, round(detected_object.depth, 2), detected_object.x, detected_object.y])
            else:
                obj.append([detected_object.class_name, round(detected_object.depth, 2)])
    else:
        print("No objects detected in the response")
    return obj

def call_whisper(): # should be implemented
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

def scan_environment(task_objects=["bottle","table"]): # add objects taken from task or give it the Task Reasonong objects
    print("VIT")
    vit_results = call_vit_service()

    prompt = "Extract all key objects(mention humans, man, woman also in objects) and environments from the Vision Transformer's scene description. Focus on common objects and spaces in a room."
    context = {"vit_results":vit_results}
    instructions = "Reason through what would make the most sense which objects/spaces are needed. Convert synonyms to common terms (e.g., 'people' to 'human') ANSWER SHORT"
    expected_response={
        "reasoning": "brief_explanation",
        "detect_objects": "[\"relevant_objects_and_humans\"]",
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
    llm_answer = call_llm_service(question=que, chat=False, reload=True)
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

def scan_environment_pipeline(goal1=30, goal2=-90):
    """
    :return: all found objects in environment
    """
    objs = []
    obj_twist = {}
    # - for right + for left
    send_rotation_goal(goal1)
    rospy.sleep(5)
    objs_scan = scan_environment()
    objs.extend(objs_scan) # use extend and not append

    detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in objs_scan]
    obj_twist = find_object_pose(detected_objects)
    #print(obj_twist)

    send_rotation_goal(goal2)
    rospy.sleep(5)
    objs_scan = scan_environment()
    objs.extend(objs_scan)

    detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in objs_scan]
    # detected_objects = [["Human","5m away"], ["Beer","8m away"], ["Kitchen","5m away"]]
    obj_twist.update(find_object_pose(detected_objects))
    #print(obj_twist)

    #print(objs)
    return objs, obj_twist # ["","",...]

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

    Parameters:
        polygon: List of points defining the polygon (numpy array of shape Nx2).
        point_inside: A point inside the polygon (numpy array of shape 2).
        buffer_distance: The distance to extend outward from the nearest edge.

    Returns:
        buffer_adjusted_point: The point outside the polygon at the specified buffer distance.
    """
    closest_point = None
    min_distance = float('inf')
    orthogonal_vector = None

    # Iterate over the edges of the polygon
    for i in range(len(polygon)):
        edge_start = polygon[i]
        edge_end = polygon[(i + 1) % len(polygon)]  # Wrap around to form edges

        # Edge vector
        edge = edge_end - edge_start
        edge_length_squared = np.dot(edge, edge)

        # Project the point onto the edge
        if edge_length_squared == 0:  # Degenerate edge
            projected_point = edge_start
        else:
            t = np.dot(point_inside - edge_start, edge) / edge_length_squared
            t = np.clip(t, 0, 1)  # Clamp t to stay within the edge segment
            projected_point = edge_start + t * edge

        # Compute distance from the point to the edge
        distance = np.linalg.norm(point_inside - projected_point)

        # Update closest point and orthogonal vector if this edge is nearer
        if distance < min_distance:
            min_distance = distance
            closest_point = projected_point

            # Compute the orthogonal vector to the edge
            edge_normal = np.array([-edge[1], edge[0]])  # Perpendicular vector
            edge_normal = edge_normal / np.linalg.norm(edge_normal)  # Normalize
            orthogonal_vector = edge_normal

    # Adjust the closest point outward by the buffer distance
    buffer_adjusted_point = closest_point + buffer_distance * orthogonal_vector

    return buffer_adjusted_point

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
            try:
                # Remove any non-numeric characters like 'm away'
                distance = float(''.join(c for c in str(distance) if c.isdigit() or c == '.' or c == '-'))
            except ValueError:
                rospy.logwarn(f"Invalid distance value for object {obj_name}: {distance}")
                continue

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
            distance = round(math.sqrt(dx ** 2 + dy ** 2 + dz ** 2), 1)

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

def navigate_to_point(target_pose=PoseStamped(), buffer_distance=0.5):
    """
    Determines the navigation point based on the target's relation to tables.

    Args:
        target_pose (PoseStamped): Target point in map frame
        buffer_distance (float): Buffer distance to keep from tables or shorten drive

    When the point is in a forbidden zone, it drives to another point outside.
    When driving to a location not in a forbidden zone, it will stop buffer_distance away.
    """
    # Extract position from PoseStamped
    target_point = np.array([target_pose.pose.position.x, target_pose.pose.position.y])
    print(target_point)

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

    # Check if target point is inside table boundaries
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

        buffer_distance = 0.4  # for testing

        if distance_to_target > buffer_distance:
            # Shorten the distance by buffer_distance
            point_to_drive = robot_position + (vector_to_target / distance_to_target) * (
                        distance_to_target - buffer_distance)
            point_to_drive[1] = point_to_drive[1]  # Note: This seems redundant

        else:
            rospy.logwarn("Target is within buffer distance; navigating directly to target.")
            point_to_drive = target_point

            # Adjust y-coordinate slightly based on its sign
            if point_to_drive[1] > 0:
                point_to_drive[1] = point_to_drive[1] - 0.4
            else:
                point_to_drive[1] = point_to_drive[1] + 0.4

    # Convert point to PoseStamped
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = point_to_drive[0]
    pose.pose.position.y = point_to_drive[1]
    pose.pose.position.z = 0.0
    #pose.pose.orientation.w = 1.0  # Default forward orientation

    # works sometime when the depth can be found
    # Calculate orientation to face the original target
    dx = target_pose.pose.position.x - point_to_drive[0]
    dy = target_pose.pose.position.y - point_to_drive[1]

    # Calculate the angle using arctan2
    target_angle = math.atan2(dy, dx)

    # Convert ang
    # le to quaternion
    quaternion = quaternion_from_euler(0, 0, round(target_angle, 1))

    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]

    print(pose)
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
    start_time = time.time()
    while not rospy.is_shutdown():
        if time.time() - start_time > 30:  # Break loop if 30 seconds have passed
            print("Timeout reached. Exiting loop.")
            break

        print("Searching for object")
        object_list = call_yolo_service(obj_search)  # Get detected objects and distances
        print(object_list)
        # Check if the target object is in the detected list and within 1 meter
        for obj in object_list:
            obj_name, distance = obj
            if obj_name in obj_search:# and (distance < 1.6 or math.isnan(distance)): # problem when nan!!!
                pub.publish(stop_cmd)  # Send stop command

                while not rospy.is_shutdown(): # bad coding
                    object_list = call_yolo_service(obj_search, with_xy=True)
                    print(object_list)

                    # Check if the target object is in the detected list and within 1 meter
                    for obj in object_list:
                        obj_name, distance, x, y = obj
                        if obj_name in obj_search :#and (distance < 1.6 or math.isnan(distance)):
                            if x < 230:
                                rotate_cmd.angular.z = 0.1
                            elif x > 410:
                                rotate_cmd.angular.z = -0.1
                            else:
                                pub.publish(stop_cmd)  # Send stop command
                                return True  # Exit once the object is found within range
                            rate = rospy.Rate(3)  # 3 Hz control loop
                            # Publish rotation command
                            pub.publish(rotate_cmd)
                            rate.sleep()

        # Continue rotating if object not found
        pub.publish(rotate_cmd)
        rate.sleep()

    pub.publish(stop_cmd)
    return False

def speech_input_whisper():
    str = ""
    str = "Bring the Woman a bottle and give it to her"
    return str

def speech_output(question="Give me a Task?", lang_id='en_US', section='', key='', wait_before_speaking=0.0):
    # Create a publisher for the /tts/goal topic
    pub = rospy.Publisher('/tts/goal', TtsActionGoal, queue_size=10)

    # Create the message
    tts_goal = TtsActionGoal()

    # Set header
    tts_goal.header = Header()
    tts_goal.header.seq = 0
    tts_goal.header.stamp = rospy.Time.now()
    tts_goal.header.frame_id = ''

    # Set goal ID
    tts_goal.goal_id = GoalID()
    tts_goal.goal_id.stamp = rospy.Time.now()
    tts_goal.goal_id.id = ''

    tts_goal.goal.text.section = ''
    tts_goal.goal.text.key = ''
    tts_goal.goal.text.lang_id = ''
    tts_goal.goal.rawtext.text = question
    tts_goal.goal.rawtext.lang_id = lang_id
    tts_goal.goal.wait_before_speaking = wait_before_speaking

    # Publish the message
    rospy.sleep(1)  # Small delay to ensure connection
    pub.publish(tts_goal)

    rospy.loginfo(f"Published TTS goal: {question}")
    print(question)

def speech_input_vit_llm():
    print("start speech_input_vit_llm")

    str = ""
    parse_error = False
    get_speech = "yes"
    get_attention = "no"
    speech_question = "Give me a Task?"
    vit_results = "Was not used, startup procedure going on..."

    while not rospy.is_shutdown():
        if not parse_error:
            print("VIT")
            vit_results = call_vit_service(question="Do the humans look towards the camera?")

        prompt = "Determine if human is actively ready to interact based on visual cues. Use the results from a ViT."
        context = {"vit_results": vit_results}
        instructions = "Reason through what would make the most sense, decide if you should start speaking with them, if yes, ask them for a task. The human needs to look at you activly! ANSWER SHORT"

        expected_response = {
            "reasoning": "brief_explanation",
            "speech": "only_yes_or_no",
            "get_attention": "only_yes_or_no",
            "question": "potential_question"
        }

        # Construct the full JSON object
        question = {
            "prompt": prompt,
            "context": context,
            "instructions": instructions,
            "expected_response": expected_response
        }

        que = json.dumps(question, indent=1)

        print("LLM")
        print(que)
        if parse_error:
            llm_answer = call_llm_service(
                question= que + " ERROR: could not parse last json output, answer again with valid json! Try to just write the Json\n\n",
                reload=True, chat=False)
            parse_error = False
        else:
            llm_answer = call_llm_service(question=que, reload=True, chat=False)
        print(llm_answer)

        # Extract JSON using regex
        json_match = re.search(r'\{.*\}', llm_answer, re.DOTALL)

        if json_match:
            json_string = json_match.group()
            try:
                # Parse the JSON
                response = json.loads(json_string)

                # Access the elements
                reasoning = response["reasoning"]
                get_speech = response["speech"]
                get_attention = response["get_attention"]
                speech_question = response["question"]

            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                parse_error = True
                continue

        if get_speech == "yes":
            speech_output(question=speech_question)
            str = speech_input_whisper()
            break
        if get_attention == "yes":
            speech_output(question=speech_question)
            continue
    return str

def get_camera_pose():
    """
    Retrieve the camera pose transform from base_link to xtion_rgb_frame.

    Returns:
        PoseStamped: Camera pose in base_link frame
    """
    # Create TF buffer and listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # Wait briefly to ensure TF is ready
    rospy.sleep(1.0)

    # Lookup transform
    transform = tf_buffer.lookup_transform(
        target_frame='base_link',
        source_frame='xtion_rgb_frame',
        time=rospy.Time(0),
        timeout=rospy.Duration(5.0)
    )

    # Create PoseStamped from transform
    camera_pose = PoseStamped()
    camera_pose.header = transform.header
    camera_pose.pose.position = transform.transform.translation
    camera_pose.pose.orientation = transform.transform.rotation

    return camera_pose

def send_torso_goal(height, duration):
    """
    Sends a torso lift goal to the torso controller.
    """
    try:
        # Create a publisher for the torso controller
        pub = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=10)

        # Wait to ensure the publisher is ready
        rospy.sleep(3)

        # Create JointTrajectory message
        trajectory = JointTrajectory()

        # Set up header
        trajectory.header = Header()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = ''

        # Set joint names
        trajectory.joint_names = ['torso_lift_joint']

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [height]  # Set desired height
        point.velocities = []  # Optional: can specify velocities if needed
        point.accelerations = []  # Optional: can specify accelerations if needed
        point.effort = []  # Optional: can specify effort if needed
        point.time_from_start = rospy.Duration(duration)  # 10 seconds to reach the goal

        # Add point to trajectory
        trajectory.points.append(point)

        # Publish the trajectory
        pub.publish(trajectory)

        return True

    except Exception as e:
        rospy.logerr(f"Error sending torso goal: {e}")
        return False

def send_gripper_goal(x, y, z):

    return False

def start_wbc():
    # Wait for services to become available
    rospy.wait_for_service('/controller_manager/switch_controller')
    rospy.wait_for_service('/controller_manager/unload_controller')

    # Switch controllers
    try:
        rospy.wait_for_service('/controller_manager/switch_controller')
        switch_controller_service = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

        # Construct request directly
        switch_controller_request = {
            "start_controllers": ['whole_body_kinematic_controller'],
            "stop_controllers": ['head_controller', 'arm_left_controller', 'arm_right_controller', 'torso_controller'],
            "strictness": 0  # STRICT
        }

        # Call service
        switch_controller_service(**switch_controller_request)
        print("switched_controller")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to switch controllers: {e}")
        return
    #time.sleep(5)
    # Unload controller
    try:
        rospy.wait_for_service('/controller_manager/unload_controller')
        unload_controller_service = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)

        # Call service directly with required argument
        unload_controller_service(name='whole_body_kinematic_controller')
        print("unload_controller")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to unload controller: {e}")
        return
    #time.sleep(5)
    # Launch tiago_dual_wbc
    try:

        # Suppress subprocess output by redirecting stdout and stderr
        proc1 = subprocess.Popen(['roslaunch', 'tiago_dual_wbc', 'tiago_dual_wbc.launch'],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        print("tiago_dual_wbc.launch")
        time.sleep(10)
        proc2 = subprocess.Popen(['roslaunch', 'tiago_dual_wbc', 'push_reference_tasks.launch',
                          'source_data_arm:=topic_reflexx_typeII',
                          'source_data_gaze:=topic'],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        print("push_reference_tasks.launch")
        time.sleep(10)
    except Exception as e:
        rospy.logerr(f"Failed to launch tiago_dual_wbc: {e}")
        return
    print("wbc_started")
    return proc1, proc2

def end_wbc(proc1, proc2):
    """
    Stops the whole_body_kinematic_controller and starts individual controllers
    for arms and torso.
    """
    # Stop proc1
    os.kill(proc1.pid, signal.SIGTERM)
    print(f"Stopped process with PID {proc1.pid}")

    # Stop proc2
    os.kill(proc2.pid, signal.SIGTERM)
    print(f"Stopped process with PID {proc2.pid}")

    try:
        # Wait for the service to be available
        rospy.wait_for_service('/controller_manager/switch_controller')

        # Create a service proxy
        switch_controller_service = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

        # Define controllers to start and stop
        start_controllers = ['arm_left_controller', 'arm_right_controller', 'torso_controller']
        stop_controllers = ['head_controller', 'whole_body_kinematic_controller']

        # Call the service
        response = switch_controller_service(
            start_controllers=start_controllers,
            stop_controllers=stop_controllers,
            strictness=0  # BEST_EFFORT mode
        )

        if response.ok:
            rospy.loginfo("Successfully switched controllers.")
        else:
            rospy.logerr("Failed to switch controllers.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call to switch_controller failed: {e}")
    except rospy.ROSException as e:
        rospy.logerr(f"Timeout waiting for switch_controller service: {e}")
    print("wbc_stopped")

def goal_wbc(x, y, z):
    """
    Sends a goal pose to the whole body kinematic controller.
    """

    # Initialize ROS node if not already initialized
    if not rospy.get_node_uri():
        rospy.init_node('goal_wbc_node', anonymous=True)

    # Publisher for the goal pose
    pub = rospy.Publisher('/whole_body_kinematic_controller/arm_left_tool_link_goal', PoseStamped, queue_size=10)

    rospy.sleep(1)  # Allow publisher to initialize

    # Create and populate PoseStamped message
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = 'base_link'
    pose_msg.pose.position.x = x
    pose_msg.pose.position.y = y
    pose_msg.pose.position.z = z
    pose_msg.pose.orientation.x = 0.7071068
    pose_msg.pose.orientation.y = 0.0
    pose_msg.pose.orientation.z = 0.0
    pose_msg.pose.orientation.w = 0.7071068

    # Publish the pose multiple times to ensure the controller receives it
    rate = rospy.Rate(10)  # 10 Hz
    for _ in range(10):
        pose_msg.header.stamp = rospy.Time.now()
        pub.publish(pose_msg)
        rate.sleep()

    # Optional sleep to allow the controller to act on the command
    time.sleep(10)

def find_point_camera_coord__(x_pixel, y_pixel, distance):
    """
    Calculate the 3D position of a point in the robot's coordinate frame.

    Args:
        x_pixel (float): The X pixel coordinate of the point.
        y_pixel (float): The Y pixel coordinate of the point.
        distance (float): The distance from the camera along the optical axis in meters.

    Returns:
        Point: The position of the point in the robot's coordinate frame.
    """
    # Camera intrinsic parameters
    K = [522.1910329546544, 0.0, 320.5,
         0.0, 522.1910329546544, 240.5,
         0.0, 0.0, 1.0]

    # Extract fx, fy, cx, cy from K
    fx = K[0]  # Focal length in X direction
    fy = K[4]  # Focal length in Y direction
    cx = K[2]  # Principal point x-coordinate
    cy = K[5]  # Principal point y-coordinate

    # Calculate the 3D coordinates in the camera frame
    x_camera = distance  # X coordinate in camera frame (along the optical axis)
    y_camera = (y_pixel - cy) * distance / fy  # Y coordinate in camera frame
    z_camera = (x_pixel - cx) * distance / fx  # Z coordinate in camera frame

    # Transform to the robot's coordinate frame
    x_robot = x_camera  # X in robot frame is X in camera frame
    y_robot = -z_camera  # Y in robot frame is -Z in camera frame
    z_robot = -y_camera  # Z in robot frame is Y in camera frame

    # Create the Point object for the robot's coordinate frame
    object_position_in_robot_frame = Point(x_robot, y_robot, z_robot)
    return object_position_in_robot_frame

def find_point_camera_coord(x_pixel, y_pixel, distance):
    """
    Calculate the 3D position of a point in the camera's coordinate frame.
    Args:
        x_pixel (float): The X pixel coordinate of the point (increasing left to right).
        y_pixel (float): The Y pixel coordinate of the point (increasing top to bottom).
        distance (float): The distance from the camera along the optical axis in meters.
    Returns:
        Point: The position of the point in the camera's coordinate frame.
    """
    # Camera intrinsic parameters
    K = [522.1910329546544, 0.0, 320.5,
         0.0, 522.1910329546544, 240.5,
         0.0, 0.0, 1.0]

    # Extract fx, fy, cx, cy from K
    fx = K[0]  # Focal length in X direction
    fy = K[4]  # Focal length in Y direction
    cx = K[2]  # Principal point x-coordinate
    cy = K[5]  # Principal point y-coordinate

    # Calculate the 3D coordinates in the camera frame
    z_camera = distance  # Z coordinate is the distance from the camera
    x_camera = (x_pixel - cx) * distance / fx  # X coordinate (horizontal, left to right)
    y_camera = -(y_pixel - cy) * distance / fy  # Y coordinate (vertical, top to bottom, negated)

    # Create the Point object for the camera's coordinate frame
    return Point(x_camera, y_camera, z_camera)

def transform_camera_to_base_link(camera_pose, point_in_camera_frame):
    """
    Transform a point from camera frame to base_link frame.

    Args:
        camera_pose (PoseStamped): Pose of the camera in base_link frame
        point_in_camera_frame (Point): Point coordinates in camera frame

    Returns:
        Point: Point coordinates in base_link frame
    """
    # Extract camera pose
    camera_x = camera_pose.pose.position.x
    camera_y = camera_pose.pose.position.y
    camera_z = camera_pose.pose.position.z

    # Apply translation
    object_position_in_base_link = Point(
        camera_x + point_in_camera_frame.z,  # camera z becomes base_link x
        camera_y - point_in_camera_frame.x,  # camera x becomes base_link y (negated)
        camera_z - point_in_camera_frame.y - 0.2   # camera y becomes base_link z (negated)
    )

    return object_position_in_base_link

def grasp(obj_search=["bottle"]):

    is_middle = False
    distance = 0.7
    send_torso_goal(0, 5)
    x = y = 0
    distance = 0.8

    for i in [x * 0.01 for x in range(31)]:
        send_torso_goal(i, 5)
        time.sleep(2)

        object_list = call_yolo_service(obj_search, with_xy=True)
        print(object_list)

        # Check if the target object is in the detected list and within 1 meter
        for obj in object_list:
            obj_name, distance, x, y = obj
            if obj_name in obj_search:# and (distance < 1.6 or math.isnan(distance)):
                if y > 240:
                    is_middle = True
        if is_middle:
            print("is in middle")
            break
    """
    # Get the camera's pose in the base_link frame
    camera_pose_base_link = get_camera_pose()

    # Assume the object is along the camera's x-axis in its frame
    # Camera frame (xtion_rgb_frame) axes: x (forward), y (right), z (down)
    # Let's place the object at a distance from the camera along its x-axis
    #object_position_in_camera_frame = Point(distance, 0, 0)  # Object at 'distance' along the camera's x-axis
    
    print(x)
    print(y)
    object_position_in_camera_frame = find_point_camera_coord(x,y,distance)
    print(object_position_in_camera_frame)
    # Now we need to transform this object position into the base_link frame
    # In this case, we directly add the camera's position to the object position:
    #object_position_in_camera_frame = Point(distance, 0, 0)
    object_position_in_base_link = Point(
        camera_pose_base_link.pose.position.x + object_position_in_camera_frame.x,
        camera_pose_base_link.pose.position.y + object_position_in_camera_frame.y,
        camera_pose_base_link.pose.position.z# + object_position_in_camera_frame.z
    )
    """
    camera_pose_base_link = get_camera_pose()
    point_in_camera_frame = find_point_camera_coord(x, y, distance)
    object_position_in_base_link = transform_camera_to_base_link(camera_pose_base_link, point_in_camera_frame)

    # Print the object's position in the base_link frame
    print(f"Object position in base_link frame: x={object_position_in_base_link.x}, "
          f"y={object_position_in_base_link.y}, z={object_position_in_base_link.z}")

    proc1, proc2 = start_wbc()
    _release_service = rospy.ServiceProxy('/parallel_gripper_left_controller/release', Empty)
    _release_service()

    goal_wbc(object_position_in_base_link.x-0.1, object_position_in_base_link.y, object_position_in_base_link.z+0.2)
    time.sleep(2)
    goal_wbc(object_position_in_base_link.x, object_position_in_base_link.y, object_position_in_base_link.z+0.2)
    time.sleep(2)
    close_gripper()

    # close the gripper
    _grasp_service = rospy.ServiceProxy('/parallel_gripper_left_controller/grasp', Empty)
    _grasp_service()

    goal_wbc(0.3, -0.3, 1)
    
    end_wbc(proc1, proc2)
    #close_gripper()
    return False

def close_gripper():

    # Create a publisher for the /parallel_gripper_left_controller/command topic
    pub = rospy.Publisher('/parallel_gripper_left_controller/command', JointTrajectory, queue_size=10)

    # Create the JointTrajectory message
    joint_traj = JointTrajectory()

    # Header information
    header = Header()
    header.seq = 0
    header.stamp = rospy.Time(0)  # Time should be set to 0 for now
    header.frame_id = ''

    # Set header
    joint_traj.header = header

    # Set joint names
    joint_traj.joint_names = ['parallel_gripper_joint']

    # Create the point
    point = JointTrajectoryPoint()
    point.positions = [0.0]  # Assuming 0 is the closed position
    point.velocities = []
    point.accelerations = []
    point.effort = []
    point.time_from_start = rospy.Duration(1.0)  # 1 second to close

    # Add point to the trajectory
    joint_traj.points.append(point)

    # Publish the message
    rospy.loginfo("Publishing JointTrajectory message to close the gripper")
    pub.publish(joint_traj)
    time.sleep(5)

def open_gripper():

    # Create a publisher for the /parallel_gripper_left_controller/command topic
    pub = rospy.Publisher('/parallel_gripper_left_controller/command', JointTrajectory, queue_size=10)

    # Create the JointTrajectory message
    joint_traj = JointTrajectory()

    # Header information
    header = Header()
    header.seq = 1
    header.stamp = rospy.Time(0)  # Time should be set to 0 for now
    header.frame_id = ''

    # Set header
    joint_traj.header = header

    # Set joint names
    joint_traj.joint_names = ['parallel_gripper_joint']

    # Create the point
    point = JointTrajectoryPoint()
    point.positions = [1.0]  # Assuming 0 is the closed position
    point.velocities = []
    point.accelerations = []
    point.effort = []
    point.time_from_start = rospy.Duration(1.0)  # 1 second to close

    # Add point to the trajectory
    joint_traj.points.append(point)

    # Publish the message
    rospy.loginfo("Publishing JointTrajectory message to open the gripper")
    pub.publish(joint_traj)
    time.sleep(5)

## I tried unstructured but it is trash
def test(): # add a History of some past taken actions and add a time to them
    rospy.init_node('prefrontal_cortex_node', anonymous=True)
    # Initialize the ROS node
    rospy.loginfo("Node started and will call services in a loop.")

    print("Start... \n\n")
    # fine_positioning()
    #navigate_to_point()
    #scan_environment_pipeline()

    #sys.exit()
    #print(aquire_task())
    #print(scan_environment())
    #print(call_vit_service())

    past_reasoning = ""
    is_start = True
    # Define variables for different sections
    modes = {
        "SPEECH": "Only possible nearby a Human below 2m distance",
        "SCAN": "Scan environment fo find known_objects",
        "NAVIGATE": "Move to known_objects",
        "PICKUP": "Grab known_object, location_robot must be at known_object",
        "PLACE": "Put down holding_object"
    }

    prompt = "You control a robot by selecting its next operating mode based on the current context. **You try to do the task that is mentioned** Start with the last mode in HISTORY, or START MODE if empty."
    prompt = "You control a robot by selecting its next operating mode based on the current context. **You try to do the task that is mentioned**"
    instructions = "Reason through what would make the most sense, create a plan if all prerequisites for the whished mode are done.  ANSWER SHORT"  # "Reason through what would make the most sense, create a plan if all prerequisites for the whished mode are done.  ANSWER SHORT"
    instructions = "REPLY IN expected_response JSON FORMAT, ANSWER SUPER SHORT, before writing the JSON, make a CHAIN-OF-THOUGHT of your plan and why it makes sense, consider the distances"
    #instructions = "Reply in expected_response JSON format. Before the JSON, analyze mode-object interactions, mode order, and distances. Keep the JSON brief."
    expected_response = {
        "reasoning": "brief_explanation_based_on_the_task",
        "next_mode": "selected_mode",
        "target_object": "if_navigation_needed",  # if_navigation_needed
    }

    # Variables for context
    robot_current_task = "Find a human an ask him"
    recent_mode = "None"  # Empty history
    detected_objects = [["None", "None"]]  # Empty known objects list
    current_location = "not_near_any_object"  # No specific location
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

        que = "PAST REASONING\n" + past_reasoning + "\nCURRENT INFORMATION\n" + json.dumps(question, indent=1)
        print(que)
        if parse_error:
            llm_answer = call_llm_service(
                question="ERROR: could not parse json, the \"next_mode\" and \"reasoning\", answer again with valid json! Try to just write the Json\n\n" + que,
                reload=True, chat=False)
            parse_error = False
        else:
            # que = json.dumps("{\n\"prompt\": \"" + question["prompt"])+"\n\n"+json.dumps(question["context"])+"\n\n"+json.dumps(question["expected_response"])
            # que = json.dumps(question)
            # print(formatted_json)
            llm_answer = call_llm_service(question=que, reload=True, chat=False)
        # print(que)
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

                #past_reasoning = past_reasoning + " reasoning: " + reasoning + "\n"

                #past_reasoning = call_llm_service(question="Summarise, SUPERSHORT and precise:\n\n" + past_reasoning, reload=True, chat=False)

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
            robot_current_task = "Bring the Human a Beer and place it down"
            # Add logic to acquire a new task
        elif next_mode == "SCAN":
            print("Scanning the environment...")
            detected_objects = [["Human","5m away"], ["Beer","8m away"], ["Kitchen","5m away"]]
            # Add logic to scan the environment
        elif next_mode == "NAVIGATE":
            print("Navigating to the specified object...")
            current_location = target_object  # No specific location
            if target_object =="Human":
                detected_objects = [["Human", "0.3m away"], ["Beer", "8m away"], ["Kitchen", "5m away"]]
            elif target_object =="Kitchen":
                detected_objects = [["Human", "5m away"], ["Beer", "8m away"], ["Kitchen", "0.3m away"]]
            elif target_object =="Beer":
                detected_objects = [["Human", "5m away"], ["Beer", "0.3m away"], ["Kitchen", "5m away"]]
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

def main_real():
    rospy.init_node('prefrontal_cortex_node', anonymous=True)

    # Initialize the ROS node
    rospy.loginfo("Node started and will call services in a loop.")
    all_found_objects_with_twist = {}
    print("Start... \n\n")

    #speech_input_vit_llm()
    #sys.exit()
    open_gripper()

    proc1, proc2 = start_wbc()
    end_wbc(proc1, proc2)

    print("wait")
    time.sleep(1)

    open_gripper()
    sys.exit()
    send_torso_goal(0.1, 5)
    time.sleep(2)
    # time.sleep(4)

    # start_wbc()
    # time.sleep(10)
    # goal_wbc(0.8, 0, 1)
    # time.sleep(10)

    grasp(obj_search=["bottle"])
    sys.exit()

    obj = []
    # - for right + for left
    # obj.extend(scan_environment())  # use extend and not append
    detected_objects = call_yolo_service(["bottle"])

    print(detected_objects)
    detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]
    print(detected_objects)

    if len(all_found_objects_with_twist) == 0:
        all_found_objects_with_twist = find_object_pose(detected_objects)
    else:
        all_found_objects_with_twist.update(find_object_pose(detected_objects))
    print(all_found_objects_with_twist["bottle"])

    navigate_to_point(target_pose=all_found_objects_with_twist["bottle"])
    rospy.sleep(10)
    fine_positioning(obj_search=["bottle"])
    grasp(obj_search=["bottle"])

    sys.exit()

    print("Scanning the environment...")
    detected_objects, detected_objects_twist = scan_environment_pipeline(goal1=-90, goal2=90)  # [["Human","5"], ["Beer","8"], ["Kitchen","5"]...]
    print(detected_objects)
    # Transform the data to match the required style
    detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]
    print(detected_objects)
    sys.exit()

    mode_feedback = ""
    is_start = True

    # Define variables for different sections
    modes = {
        "SPEECH": "Only possible nearby a Human below 2m distance",
        "SCAN": "Scan environment fo find known_objects",
        "NAVIGATE": "Move to known_objects",
        "PICKUP": "Grab known_object, needs to be below 1.5m distance",
        "PLACE": "Put down holding_object"
    }

    prompt = "You control a robot by selecting its next operating mode based on the current context. **You try to do the task that is mentioned**"
    instructions = "REPLY IN expected_response JSON FORMAT, ANSWER SUPER SHORT, before writing the JSON, make a CHAIN-OF-THOUGHT of your plan and why it makes sense, consider the distances and the implications/shortcomings your choices have."

    expected_response = {
        "reasoning": "brief_explanation_based_on_the_task",
        "next_mode": "selected_mode",
        "target_object": "None or navigation or specific_scan_object",  # if_navigation_needed
    }

    # Variables for context
    robot_current_task = "Find a human an ask him"
    recent_mode = "None"  # Empty history
    detected_objects = [["None", "None"]]  # Empty known objects list
    current_location = "not_near_any_object"  # No specific location
    current_held_object = "None"  # Not holding anything

    parse_error = False
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        context = {
            "modes": modes,
            "task": robot_current_task,
            "modes_history": recent_mode,
            "known_objects": detected_objects,
            "location_robot": current_location,
            "holding_object": current_held_object,
        }

        # Construct the full JSON object
        question = {
            "prompt": prompt,
            "context": context,
            "instructions": instructions,
            "mode_feedback": mode_feedback,
            "expected_response": expected_response
        }

        que = json.dumps(question, indent=1)
        mode_feedback = ""

        print(que)
        if parse_error:
            llm_answer = call_llm_service(
                question="ERROR: could not parse json, the \"next_mode\" and \"reasoning\", answer again with valid json! Try to just write the Json\n\n" + que,
                reload=True, chat=False)
            parse_error = False
        else:
            llm_answer = call_llm_service(question=que, reload=True, chat=False)
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

                # past_reasoning = past_reasoning + "reasoning: " + reasoning + "\n"

                print(f"Reasoning: {reasoning}")
                print(f"Next mode: {next_mode}")
                print(f"Target object: {target_object}")

            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                next_mode = ""
        else:
            next_mode = ""
            print("No JSON object found in the response.")

        if next_mode == "SPEECH":


            print("Acquiring a new task...")
            robot_current_task = speech_input()

        elif next_mode == "SCAN":  # detected_objects_twist is bad coding
            if target_object != "None":
                print("Scanning the environment for specific object")
                if False or fine_positioning(obj_search=[target_object]):
                    objs_scan = call_yolo_service(detect_objects)

                    dec_obj = [[obj[0], f"{obj[1]}m away"] for obj in objs_scan]
                    detected_objects.append(dec_obj)
                    # detected_objects = [["Human","5m away"], ["Beer","8m away"], ["Kitchen","5m away"]]
                    all_found_objects_with_twist.update(find_object_pose(dec_obj))
                    continue
                else:
                    mode_feedback = "LAST MODE SCAN COULD NOT FIND " + target_object + " AT CURRENT LOACTION"
                    continue

            print("Scanning the environment...")
            detected_objects, detected_objects_twist = scan_environment_pipeline()  # [["Human","5"], ["Beer","8"], ["Kitchen","5"]...]

            # Transform the data to match the required style
            detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]

            # print(detected_objects)
            # detected_objects = [["Human","5m away"], ["Beer","8m away"], ["Kitchen","5m away"]]

            if len(all_found_objects_with_twist) == 0:
                all_found_objects_with_twist = detected_objects_twist  # this is wrong is should be done after every scan before
            else:
                all_found_objects_with_twist.update(detected_objects_twist)

        elif next_mode == "NAVIGATE":  # make a difference from space to object, cant drive to near to space but can
            # drive near to object
            print("Navigating to the specified object...")
            navigate_to_point(target_pose=all_found_objects_with_twist[target_object])

            rospy.sleep(10)  # look if it does stop then

            send_torso_goal(0.1, 5)
            rospy.sleep(2)
            if target_object != "kitchen":
                fine_positioning(obj_search=[target_object])
            current_location = target_object  # No specific location
            detected_objects = objects_pose_to_distance(all_found_objects_with_twist)
            detected_objects_no_style = detected_objects
            detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]

        elif next_mode == "PICKUP":
            for obj in detected_objects_no_style:
                object_name, distance = obj
                # Check if the object is the target and within 1.5 meters
                if object_name == target_object:
                    if distance <= 1.5:
                        # Object is nearby, attempt to grasp
                        grasp(obj_search=[target_object])
                        current_held_object = target_object
                        print(f"Picking up the {target_object}...")
                        break
                    else:
                        # Object is too far
                        env_feedback = f"Object can't be grasped because it is {distance}m away (too far)"
                        print(env_feedback)
                        break
            else:
                # No target object found
                env_feedback = "Object not detected in the environment"
                print(env_feedback)

        elif next_mode == "PLACE":
            print("Placing the object...")
            open_gripper()
            break

        else: # Handle unexpected mode
            print(f"Unknown mode: {next_mode}")
            parse_error = True

        # Variables for context
        recent_mode = next_mode

        rospy.loginfo("All services have been called in this cycle.")
        rate.sleep()

def main_virtual(): # add a History of some past taken actions and add a time to them
    rospy.init_node('prefrontal_cortex_node', anonymous=True)
    # Initialize the ROS node
    rospy.loginfo("Node started and will call services in a loop.")
    all_found_objects_with_twist = {}
    print("Start... \n\n")
    open_gripper()

    proc1, proc2 = start_wbc()
    end_wbc(proc1, proc2)
    print("wait")
    time.sleep(1)

    open_gripper()
    send_torso_goal(0.1, 5)
    time.sleep(2)
    #time.sleep(4)

    #start_wbc()
    #time.sleep(10)
    #goal_wbc(0.8, 0, 1)
    #time.sleep(10)

    #grasp(obj_search=["bottle"])
    #sys.exit()

    obj = []
    # - for right + for left
    #obj.extend(scan_environment())  # use extend and not append
    detected_objects = call_yolo_service(["bottle"])

    print(detected_objects)
    detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]
    print(detected_objects)

    if len(all_found_objects_with_twist) == 0:
        all_found_objects_with_twist = find_object_pose(detected_objects)
    else:
        all_found_objects_with_twist.update(find_object_pose(detected_objects))
    print(all_found_objects_with_twist["bottle"])

    navigate_to_point(target_pose=all_found_objects_with_twist["bottle"])
    rospy.sleep(10)
    fine_positioning(obj_search=["bottle"])
    grasp(obj_search=["bottle"])

    sys.exit()



    print("Scanning the environment...")
    detected_objects, detected_objects_twist = scan_environment_pipeline() # [["Human","5"], ["Beer","8"], ["Kitchen","5"]...]
    print(detected_objects)
    # Transform the data to match the required style
    detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]
    print(detected_objects)
    sys.exit()


    mode_feedback = ""
    is_start = True

    modes = {
        "SPEECH": "Only possible nearby a Human below 2m distance",
        "SCAN": "Scan environment fo find known_objects",
        "NAVIGATE": "Move to known_objects",
        "PICKUP": "Grab known_object, needs to be below 1.5m distance",
        "PLACE": "Put down holding_object"
    }

    prompt = "You control a robot by selecting its next operating mode based on the current context. **You try to do the task that is mentioned**"
    instructions = "REPLY IN expected_response JSON FORMAT, ANSWER SUPER SHORT, before writing the JSON, make a CHAIN-OF-THOUGHT of your plan and why it makes sense, consider the distances and the implications/shortcomings your choices have."

    expected_response = {
        "reasoning": "brief_explanation_based_on_the_task",
        "next_mode": "selected_mode",
        "target_object": "None or navigation or specific_scan_object",  # if_navigation_needed
    }

    # Variables for context
    robot_current_task = "Find a human an ask him"
    recent_mode = "None"  # Empty history
    detected_objects = [["None", "None"]]  # Empty known objects list
    current_location = "not_near_any_object"  # No specific location
    current_held_object = "None"  # Not holding anything

    parse_error = False
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        context = {
            "modes": modes,
            "task": robot_current_task,
            "modes_history": recent_mode,
            "known_objects": detected_objects,
            "location_robot": current_location,
            "holding_object": current_held_object,
        }

        # Construct the full JSON object
        question = {
            "prompt": prompt,
            "context": context,
            "instructions": instructions,
            "mode_feedback": mode_feedback,
            "expected_response": expected_response
        }

        que = json.dumps(question, indent=1)
        mode_feedback = ""
        print(que)
        if parse_error:
            llm_answer = call_llm_service(
                question="ERROR: could not parse json, the \"next_mode\" and \"reasoning\", answer again with valid json! Try to just write the Json\n\n" + que,
                reload=True, chat=False)
            parse_error = False
        else:
            llm_answer = call_llm_service(question=que, reload=True, chat=False)
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

                # past_reasoning = past_reasoning + "reasoning: " + reasoning + "\n"

                print(f"Reasoning: {reasoning}")
                print(f"Next mode: {next_mode}")
                print(f"Target object: {target_object}")

            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                next_mode = ""
        else:
            next_mode = ""
            print("No JSON object found in the response.")

        if next_mode == "SPEECH":
            print("Acquiring a new task...")
            robot_current_task = speech_input()

        elif next_mode == "SCAN": #detected_objects_twist is bad coding
            if target_object != "None":
                print("Scanning the environment for specific object")
                if False or fine_positioning(obj_search=[target_object]):
                    objs_scan = call_yolo_service(detect_objects)

                    dec_obj = [[obj[0], f"{obj[1]}m away"] for obj in objs_scan]
                    detected_objects.append(dec_obj)
                    # detected_objects = [["Human","5m away"], ["Beer","8m away"], ["Kitchen","5m away"]]
                    all_found_objects_with_twist.update(find_object_pose(dec_obj))
                    continue
                else:
                    mode_feedback = "LAST MODE SCAN COULD NOT FIND "+ target_object +" AT CURRENT LOACTION"
                    continue

            print("Scanning the environment...")
            detected_objects, detected_objects_twist = scan_environment_pipeline() # [["Human","5"], ["Beer","8"], ["Kitchen","5"]...]

            # Transform the data to match the required style
            detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]

            #print(detected_objects)
            #detected_objects = [["Human","5m away"], ["Beer","8m away"], ["Kitchen","5m away"]]

            if len(all_found_objects_with_twist) == 0:
                all_found_objects_with_twist = detected_objects_twist # this is wrong is should be done after every scan before
            else:
                all_found_objects_with_twist.update(detected_objects_twist)

        elif next_mode == "NAVIGATE": # make a difference from space to object, cant drive to near to space but can
            # drive near to object
            print("Navigating to the specified object...")
            navigate_to_point(target_pose=all_found_objects_with_twist[target_object])

            rospy.sleep(10) # look if it does stop then

            send_torso_goal(0.1, 5)
            rospy.sleep(2)
            if target_object != "kitchen":
                fine_positioning(obj_search=[target_object])
            current_location = target_object  # No specific location
            detected_objects = objects_pose_to_distance(all_found_objects_with_twist)
            detected_objects_no_style = detected_objects
            detected_objects = [[obj[0], f"{obj[1]}m away"] for obj in detected_objects]

        elif next_mode == "PICKUP":
            for obj in detected_objects_no_style:
                object_name, distance = obj

                # Check if the object is the target and within 1.5 meters
                if object_name == target_object:
                    if distance <= 1.5:
                        # Object is nearby, attempt to grasp
                        grasp(obj_search=[target_object])
                        current_held_object = target_object
                        print(f"Picking up the {target_object}...")
                        break
                    else:
                        # Object is too far
                        env_feedback = f"Object can't be grasped because it is {distance}m away (too far)"
                        print(env_feedback)
                        break
            else:
                # No target object found
                env_feedback = "Object not detected in the environment"
                print(env_feedback)

        elif next_mode == "PLACE":
            print("Placing the object...")
            open_gripper()
            break

        else: # Handle unexpected mode
            print(f"Unknown mode: {next_mode}")
            parse_error = True

        # mode
        recent_mode = next_mode

        rospy.loginfo("All services have been called in this cycle.")
        rate.sleep()

if __name__ == '__main__':
    try:
        # test()
        #main_virtual()
        main_real()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
