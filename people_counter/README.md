This project tracks persons on the video
## Repository description
- `input` - folder for input videos
- `output` - folder for output videos
- `people_tracker` - the main code 
	- `yolo` - definition of the YOLOv3 model
	- `detector_yolo.py` - initialize yolo detector
	- `draw_detections.py` - draw detections on an image
	- `main.py` - track people
	- `tools.py` - utils 
	- `tracker_cv2.py` - cv2 KCF tracker 
	- `tracker_dlib.py` - Dlib correlation tracker
	- `canvas.py` - defines zones for camera
 - `Dockerfile`
- `environment.yml`

## Problem
The goal is to count how many people are in the shop.

## Solution
To do this:
Track persons alternating detection (yolo) and tracking (KCF cv2) phases. When the person change the zone increase or decrease counter.

### YOLO
[Download weights](https://github.com/wizyoung/YOLOv3_TensorFlow/releases/)
[Download weights mirror](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing)

Save downloaded files to:

`people_tracker/yolo/data/darknet_weights/` directory

Don't forget to select the downloaded model in `main.py` or `main_center.py` 

### How to run the code
Enter the docker environment
```
sudo xhost +local:root 
docker-compose -f environment.yml up --build
docker exec -it trackingpeople_tracker_1 bash
```
Then do whatever you want in the docker container:
```
cd people_tracker
python3 main_center.py ../input/video.mp4
```

## Used materials:
- [TF model zoo](https://github.com/Sarrasor/TrackingPeople): was used as a template
- [Yolo](https://github.com/wizyoung/YOLOv3_TensorFlow): yolo model with weights