DOCKER_FILE="$1"
IMAGE_NAME="$2"

docker build --tag ${IMAGE_NAME} --file ${DOCKER_FILE} .

xhost +

docker run --rm -it --privileged --gpus all \
    --device /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
    --user $(id -u):$(id -g) --mount type=bind,source="$(pwd)",target=/workspace ${IMAGE_NAME} bash
