#!/usr/bin/env python3

import rospy
from std_msgs.msg import String  # Import ROS message type for string publishing

# Global variables
is_running = True

def text_input_loop():
    """ Simulates transcription by taking text input from the terminal and publishing it. """
    global is_running
    
    # Create a ROS publisher
    pub = rospy.Publisher('speech_recognition/transcription', String, queue_size=10)
    
    rospy.loginfo("Text input mode. Type your text to transcribe. Type 'exit' to quit.")
    
    while not rospy.is_shutdown() and is_running:
        # Get user input from the terminal
        user_input = input("Enter text: ").strip()
        
        if user_input.lower() == 'exit':
            rospy.loginfo("Exiting transcription mode.")
            is_running = False
            break
        
        if user_input:
            # Simulate transcription by publishing the entered text to the ROS topic
            rospy.loginfo(f"Transcribed Text: {user_input}")
            pub.publish(user_input)  # Publish the transcribed text to the ROS topic

def main():
    global is_running
    
    # Initialize the ROS node
    rospy.init_node('text_transcription_node', anonymous=True)
    
    rospy.loginfo("Text transcription node started. Waiting for input...")
    
    # Start the text input loop in the main thread
    text_input_loop()
    
    # Keep running until ROS is shut down
    rospy.spin()
    
    rospy.loginfo("Shutting down text transcription node.")

if __name__ == "__main__":
    main()

