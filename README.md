# robot-manipulation-llm
This implementation allows the Tiago Robot to autonomously reason through a fetching task; by utilising compact LLMs! 

## Setup
- Set up a Tiago Docker; supplied by Pal-Robotics for the barebone or ask me for the final docker image. 
- create a catkin workspace **lmt_ws** in the **exchange** directory
- create packages [llm_fetch_me](./llm_fetch_me) and [custom_msg_srv](./custom_msg_srv) and copy the GIT data into it
- install the ollama package on the PC outside the Docker and download; Ollama communicates with the Docker over the internet ports. (CUDA does not work inside the docker)
´´´
ollama pull gemma2:9b-instruct-q8_0
´´´

## Starting the Docker

## For Debugging when problems arise with the Robot in virtual or real environment
Please open the the html file in the Browser, it is the best way to view it. There is a lot of information in it thanks! [File](./How_to_start_and_operate.html)
