# Osteoarthritis_classification

This is a software developed for Osteoarthritis classification under paper: **Automated grading of Radiographic Knee Osteoarthritis Severity combing with Joint space narrowing**. After importing a knee X-ray raw image of the Posterior-Anterior (PA) view, this software will automatically grade knee osteoarthritis severity (KOA), knee joint detection, knee bone segmentation, and a joint spacing narrowing grading.

This repository offers executable software on visualization (GUI) versions for an individual case display and a non-display performance for multiple cases processing. In addition, this executable software is user-friendly and requires no coding background and python environments; we believe these could work as a reference and learning software for radiologists' daily diagnosis.

Besides the executable version, we also offer source codes as well as dockerfile and python environment requirements. We are welcome any further developments and modifications following our non-commercial use license.
 

## run executable version of software (no python enviroment required).

### version 1: the non-dislay version
 step 1: go to the exe folder , for example: path to /image_final_pipeline
 
 step 2: type in the command line:
 ```
./image_final_pipeline --img-name --png-path --txt-path
```

notice:
--img-name: the path and the name of the dicom image. (** we also support png image version now is raw dicom file is not available)
--png-path: the path to save the result as an png image.
--txt-path: save the result in an txt file.

for exmaple:
```
./image_final_pipeline \
--img-name "102451" \
--png-path "img_folder/result.png" \
--txt-path "txt_folder/result.txt" \
```
### version 2: the GUI version
step 1： go to the exe folder, for example: /image_final_gui

step 2: type in the command line:
```
./image_final_gui
```

step 3: it will generate a window with two button on the top (The GUI is shown below, the blue contor is the segmentation results and the red lines are the joint space distance on both medial side and lateral side), then click "import" to import the dicom image.

step 4: wait for about 30 seconds, the results will show on window. 

step 5: click "save" button to choose the path to save the png and txt result. 

![9809967](https://user-images.githubusercontent.com/39239103/153900897-ad8e4ec2-f794-4674-a512-50436b383fc4.png)


## run source code under docker environment 
 step 1: setup docker environment on linux.
  run dockerfile, modify the path under run_anydevice.sh, and use command line to run the bashfile.
 ```
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
 ```
 step 2: go to the root folder.
### version 1: 
run command: 
```
python image_final_gui.py 
```
### version 2:
run: 
```
python image_final_pipeline.py  --img-name --png-path --txt-path
```

## Citation
```
```

