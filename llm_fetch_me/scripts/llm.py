#!/usr/bin/env python3

import rospy
import ollama
import re
from custom_msg_srv.srv import InfString, InfStringResponse

class LLMBrain:
    def __init__(self, model_name="llama3.2:1b-instruct-q2_K", system_prompt="", keep_alive='3m'): #minicpm-v:8b-2.6-fp16
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
            return InfStringResponse(answer=self.chat(prompt=req.question))
        return InfStringResponse(answer=self.generate_response(prompt=req.question))

    def start_service(self, nodeName='llm_inference_service', servName='llm_inference'):
        rospy.init_node(nodeName)
        s = rospy.Service(servName, InfString, self.handle_inference)
        rospy.spin()

    def generate_response(self, prompt, format="", temperature=0.0):
        #print(prompt)
        response = ollama.generate(model=self.model_name, prompt=prompt, system=self.system_prompt, keep_alive=self.keep_alive, format=format, options={"temperature": temperature})
        #print(response)
        generated_text = response['response']
        #print(generated_text)
        if "marco" in self.model_name:
            match = re.search(r"<Output>(.*?)</Output>", generated_text, re.DOTALL)
            if match:
                generated_text = match.group(1).strip()
        self.messages.append({'role': 'assistant', 'content': generated_text})
        print(generated_text)
        return generated_text
     
    def chat(self, prompt, format="", temperature=0.0):
        #print(prompt)
        self.messages.append({'role': 'user', 'content': prompt})
        
        response = ollama.chat(
            model=self.model_name,
            messages=self.messages,
            keep_alive=self.keep_alive,
            format=format)#,
            #options={"temperature": temperature}
        #)
        #print(response)
        generated_text = response['message']['content']
        #print(generated_text)
        """
        if "marco" in self.model_name:
            match = re.search(r"<Output>(.*?)</Output>", generated_text, re.DOTALL)
            if match:
                generated_text = match.group(1).strip()t
        """
        self.messages.append({'role': 'assistant', 'content': generated_text})

        #print(generated_text)
        return generated_text
        
    def reload(self):
        self.messages = []
        self.messages.append({'role': 'system', 'content': self.system_prompt})

if __name__ == "__main__":
    # Read the content of the text file
    #with open('~/exchange/lmt_ws/src/llm_fetch_me/scripts/system_prompt.txt', 'r', encoding='utf-8') as file:
        #system_prompt = file.read()
    system_prompt = "You are the brain of a robot; you make decisions based on a given task and are able to decide what to do... When thinking through a problem, you go by it direct and coldhearted, but you always think like a learner/child/discoverer; if you are not certain about a decision, you may make an educated guess. Additionally, you answer directly and understandably no blabla"

    brain = LLMBrain(model_name="gemma2:9b-instruct-q8_0") # model_name='llama3.1:latest', model_name='gemma2:9b-instruct-q8_0', marco-o1:7b-q8_0
    brain.start_service()

