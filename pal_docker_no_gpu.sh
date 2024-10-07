#!/bin/bash

# Variables required for logging as a user with the same id as the user running this script
export LOCAL_USER_ID=`id -u $USER`
export LOCAL_GROUP_ID=`id -g $USER`
export LOCAL_GROUP_NAME=`id -gn $USER`

# Allow customization for docker args
# For instance, to remove the container when it exits: DOCKER_USER_ARGS='--rm' pal_docker.sh ...
DOCKER_USER_ARGS="$DOCKER_USER_ARGS --env LOCAL_USER_ID --env LOCAL_GROUP_ID --env LOCAL_GROUP_NAME"

# Variables for forwarding ssh agent into docker container
SSH_AUTH_ARGS=""
if [ ! -z $SSH_AUTH_SOCK ]; then
    DOCKER_SSH_AUTH_ARGS="-v $SSH_AUTH_SOCK:/run/host_ssh_auth_sock -e SSH_AUTH_SOCK=/run/host_ssh_auth_sock"
fi

# Settings for X11 forwarding (if needed)
DOCKER_X11_ARGS="--env DISPLAY --env QT_X11_NO_MITSHM=1 --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"

# Basic Docker command
DOCKER_COMMAND="docker run"

DOCKER_NETWORK_ARGS="--net host"
if [[ "$@" == *"--net "* ]]; then
    DOCKER_NETWORK_ARGS=""
fi

xhost +
$DOCKER_COMMAND \
$DOCKER_USER_ARGS \
$DOCKER_X11_ARGS \
$DOCKER_SSH_AUTH_ARGS \
$DOCKER_NETWORK_ARGS \
--privileged \
-v "$HOME/exchange:/home/user/exchange" \
-v /var/run/docker.sock:/var/run/docker.sock \
"$@"
