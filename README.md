# robot-manipulation-llm
This implementation allows the Tiago Robot to autonomously reason through a fetching task; by utilising compact LLMs! 

## Setup
- Use a GPU with at least 16GB VRAM, else the pipeline runs slow, only using a CPU is not recommended except it is really strong
- Set up a Tiago Docker, supplied by [Pal-Robotics](https://docs.pal-robotics.com/sdk-dev/development/docker-public) for the barebone, or ask me for the updated Docker image.
- install the ollama package on the PC outside the Docker and download a model; Ollama communicates with the Docker over the internet access point. (CUDA does not work inside the docker)
- LLM ```ollama pull gemma2:9b-instruct-q8_0```
- VIT ```ollama pull minicpm-v:8b```
- start the docker with **pal_docker.sh** when a GPU is available or **pal_docker_no_gpu.sh** if not
  After setting up the pal docker environment, the docker can be started with 
```bash
# copy the .sh into the docker setup directory, you docker may not be named tiago-students
./pal_docker.sh -it tiago-students /bin/bash

pip install ollama tmux ultralytics nano
``` 

- create a ROS catkin workspace **lmt_ws** in the **exchange** directory
- create packages [llm_fetch_me](./llm_fetch_me) and [custom_msg_srv](./custom_msg_srv) and copy the data from the GIT into it

## Starting the virtual Pipeline

```bash
# source
source /opt/pal/gallium/setup.bash

# build the workspace
cd ~/exchange/lmt_ws
catkin_make
source ~/exchange/lmt_ws/devel/setup.bash

# when the lidar does not work in the virtual environment
export LIBGL_ALWAYS_SOFTWARE=1
```
after sourcing... when some bash lines do not work re-source in the additional tmux windows 
```bash
# create a tmux terminal
tmux new

# start virtual env, it will open GAZEBO and RVIZ; Place a human into the left corner, perspective seen from robot camera
roslaunch tiago_dual_171_gazebo tiago_dual_navigation.launch world:=pick end_effector_left:=pal-gripper end_effector_right:=pal-gripper advanced_navigation:=true

# lodad map for LIDAR localisation
rosservice call /pal_navigation_sm "input: 'LOC'"

# Change to your map, copy the map from GIT into the correct folder
rosservice call /pal_map_manager/change_map "input: '/home/user/.pal/tiago_dual_maps/configurations/map_1'"
rosservice call /global_localization "{}"
rosservice call /move_baswde/clear_costmaps "{}"

# starting the LLM, ViT and YOLO, in new tmux window with (Strg + B C); windows can be switched with (Strg + B W)
roslaunch llm_fetch_me launch_services.launch

# starting pipeline, in new tmux window with (Strg + B C)
rosrun llm_fetch_me robot_brain_node_advanced.py

```

### Starting the real Pipeline is really buggy and there is no clear way to start it. All important informations are in the next Section. 
## For Debugging when problems arise with the Robot in virtual or real environment
Please open the the html file in the Browser, it is the best way to view it. There is a lot of information in it thanks! [File](./How_to_start_and_operate.html)
