export DOCKER_DIRECTORY='path to test_folder'
export PROJECT_DIRECTORY='path to test_folder'
export DOCKER_IMG=keyu_image_new
export CONTAINER_NAME=knee_project_container_new
export DATA_FOLDER='where the test_images is'

cd $DOCKER_DIRECTORY 
docker build -t $DOCKER_IMG -f Dockerfile .
nvidia-docker run -d -it \
--name $CONTAINER_NAME \
--shm-size=14g \
-v $DOCKER_DIRECTORY:/Dockerspace \
-v $PROJECT_DIRECTORY:/workspace \
-v $DATA_FOLDER:/data \
$DOCKER_IMG

docker exec -it $CONTAINER_NAME /bin/bash



