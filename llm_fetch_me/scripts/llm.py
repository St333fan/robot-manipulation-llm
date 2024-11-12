#!/usr/bin/env python3

import rospy
import ollama
from std_srvs.srv import Trigger, TriggerResponse
from custom_msg_srv.srv import InfString, InfStringResponse

class LLMBrain:
    def __init__(self, model_name="llama3.2:1b-instruct-q2_K", system_prompt="", keep_alive='10s'): #minicpm-v:8b-2.6-fp16
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.keep_alive = keep_alive
        self.messages = []
        self.messages.append({'role': 'system', 'content': self.system_prompt})

    def handle_inference(self, req):
        print("Service call received: Starting LLM model inference")
        if req.reload:
            self.reload()
            
        if req.chat:
            return InfStringResponse(answer=self.chat(req.question))
        return InfStringResponse(answer=self.generate_response(req.question))

    def start_service(self, nodeName='llm_inference_service', servName='llm_inference'):
        rospy.init_node(nodeName)
        s = rospy.Service(servName, InfString, self.handle_inference)
        rospy.spin()

    def generate_response(self, prompt):
        response = ollama.generate(model=self.model_name, prompt=prompt, system=self.system_prompt, keep_alive=self.keep_alive)
        generated_text = response['response']
        return generated_text
     
    def chat(self, prompt):

        self.messages.append({'role': 'user', 'content': prompt})
        
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

if __name__ == "__main__":
    brain = LLMBrain(model_name='llama3.1:latest', system_prompt='You are a dog wuff wuff')
    brain.start_service()

