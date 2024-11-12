#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from custom_msg_srv.srv import InfString, InfStringResponse

def handle_inference(req):
# Print a message to indicate the service call was received
    print("Service call received: Starting AI model inference")

    # Run AI model inference
    result = req.question
    return InfStringResponse(answer=result)

def ai_inference_service():
    rospy.init_node('ai_inference_service')
    s = rospy.Service('vit_inference', InfString, handle_inference)
    rospy.spin()

if __name__ == "__main__":
    ai_inference_service()

