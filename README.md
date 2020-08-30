# sight.ai
### A visual AI assistant made for visually impaired person to navigate around the city.


# Create conda environment
```
conda env create --file environment.yml
conda activate sightai
```

# Weights
Download weights for [MonoDepth](https://u.pcloud.link/publink/show?code=XZb5r97ZD7HDDlc237BMjoCbWJVYMm0FLKcy) and [YOLOv4](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view) (see Credits).
```
place them under:
./pretrained/monodepth_resnet18_001.pth 
./pretrained/yolov4.weights
```



# Example scripts:
## run depth map
```
cd src
python demo_depth_map.py
```

## run bbox
```
python demo_yolo.py
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
<https://github.com/OniroAI/MonoDepth-PyTorch>







