# Osteoarhritis_classification

 it is a readme for dicom image kl-classification software.

## run executable version of software (no python enviroment required).

### version 1: the command line version
 step 1: go to the exe folder , for example: /image_final_pipeline
 step 2: type in the command line:
 ```
./image_final_pipeline --img-name --png-path --txt-path
```

notice:
--img-name: the path and the name of the dicom image.
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
step1： go to the exe folder, for example: /image_final_gui
step2: type in the command line:
```
./image_final_gui
```

step3: it will generate a window with two button on the top, then click "import" to import the dicom image.

step4: wait for about 30 seconds, the result will show on window.

step5: click "save" button to choose the path to save the png and txt result.

## run source code under docker environment 
 step1: setup docker environment on linux.
 step2: go to root folder.
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

