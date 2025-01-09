# robot-manipulation-llm
This implementation allows the Tiago Robot to autonomously reason through a fetching task; by utilising compact LLMs! 

## Setup
- Set up a Tiago Docker, supplied by [Pal-Robotics](https://docs.pal-robotics.com/sdk-dev/development/docker-public) for the barebone, or ask me for the updated Docker image.
- install the ollama package on the PC outside the Docker and download a model; Ollama communicates with the Docker over the internet access point. (CUDA does not work inside the docker) ```ollama pull gemma2:9b-instruct-q8_0```
- start the docker with **pal_docker.sh** when a GPU is available or **pal_docker_no_gpu.sh** if not
- create a ROS catkin workspace **lmt_ws** in the **exchange** directory
- create packages [llm_fetch_me](./llm_fetch_me) and [custom_msg_srv](./custom_msg_srv) and copy the data from the GIT into it

## Starting the Pipeline
```
# source
source /opt/pal/gallium/setup.bash
source ~/exchange/lmt_ws/devel/setup.bash

# build the workspace
cd ~/exchange/lmt_ws
catkin_make
source ~/exchange/lmt_ws/devel/setup.bash

# when the lidar does not work in the virtual environment
export LIBGL_ALWAYS_SOFTWARE=1
```
```
# create a tmux terminal
tmux new

# start virtual env

```


## For Debugging when problems arise with the Robot in virtual or real environment
Please open the the html file in the Browser, it is the best way to view it. There is a lot of information in it thanks! [File](./How_to_start_and_operate.html)
