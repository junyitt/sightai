# sight.ai
### A visual AI assistant made for visually impaired person to navigate around the city.


# Create conda environment
```
conda env create --file environment.yml
conda activate sightai
```

# Example scripts:
## run depth map
```
cd src
python depth_map.py
```

## run bbox
```
python demo.py -cfgfile cfg/yolov4.cfg -weightfile yolov4.weights -imgfile ./001_L.png
```

# Architecture
## General object detection and depth estimation module
1. img -> depth_map fn -> depthmap
2. img -> bbox_yolo fn -> bbox
3. *multiply and median to get depth score for each bbox object
4. *set a calibration factor to measure distance (might not be necessary) - we can categorize the distance into (very close, nearby, far)
5. *convert bbox and depth info into text -> speech

## QR module


# Demo on video:
https://www.youtube.com/watch?v=5H3UW2L_TlM

# Credits
### YOLOv4 pytorch implementation by Tianxiaomo
<https://github.com/Tianxiaomo/pytorch-YOLOv4>

### Unofficial implementation of Unsupervised Monocular Depth Estimation neural network MonoDepth in PyTorch by OniroAI
<https://github.com/Tianxiaomo/pytorch-YOLOv4>





