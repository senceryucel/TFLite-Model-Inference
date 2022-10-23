# TFLite Model Inference Code

### Created and coded by: <br>Sencer YÜCEL (https://github.com/senceryucel) & Behiç KILINÇKAYA (https://github.com/BehicKlncky)
<br>

### This script aims to test the performance of a TFLite Classification Model.

### MicroPython has been used in this script on OpenMV IDE. 

<br>

## Algorithm
#### It takes the .jpg image and crop it into desired number of equal parts (name of the variable to set crop number is CROP_COUNT). Then, for every mini-frame (cropped photo), TFLite Model runs and the result of classification is compared with the ground truth (.json annotation). The performance of the model is calculating based on this comparison.

<br>

## Pre-Requirities

#### - H7 Plus or or higher version of an OpenMV board.
#### - OpenMV IDE.
#### - A model to test.
#### - A testing dataset in .jpg and their annotations in .json; example format of the .json file is in the repository [Recommended: 100+ photos with different scenarios (easy-medium-hard)].

<br>

## Usage 

#### 1-) Open INFERENCE_CODE.py on OpenMV IDE, then plug your OpenMV board into your computer.

#### 2-) Construct the first lines of the script with your own paths to your own directories: 
```
PATH_TO_JSON = "PATH_TO_ANNOTATION_JSON_FILE"
PATH_TO_DATASET = "PATH_TO_YOUR_DATASET"
PATH_TO_CROPPED_PHOTOS_TO_SAVE = "PATH_TO_CROPPED_PHOTOS_TO_SAVE"
PATH_TO_TFLITE_MODEL = "PATH_TO_MODEL.tflite"
PATH_TO_SAVE_INFERENCE_RESULTS = "INFERENCE_RESULTS.txt"
CROP_COUNT = 16
```
```
What CROP_COUNT is has been described in the Algorithm part.
```

#### 3-) Connect your OpenMV Board to the IDE with Ctrl+E and execute the script with Ctrl+R.

#### 4-) You have your output in txt format to the directory you set with the information [accuracy, precision, recall, F1_score] of your TFLite model.
