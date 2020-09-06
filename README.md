# sight.ai
### A visual AI assistant made for visually impaired person to navigate around the city.


## Create conda environment
```
conda env create --file environment.yml
conda activate sightai
```

## Download Model Weights
Download weights for [BTS](https://drive.google.com/file/d/1_mENn0G9YlLAAr3N8DVDt4Hk2SBbo1pl/view) and [YOLOv4](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view) (see Credits).
```
place them under:
./pretrained/bts_latest
./pretrained/yolov4.weights
```

## Architecture
### General depth estimation module
1. Image -> Infer Depth Map
2. Depth Map -> Calculate mean on Depth Map (for each grid area) to produce Vibrational Mappings

## Example scripts:
### Run demo on sample images
```
python run_demo_image.py -input media/165_R.png -plot 1
```
Output can be found at src/output/_165_R.png

### Run demo on stairs.mp4
```
python run_demo_video.py -input media/two_way.mp4 -fps 30.0 -max 20
```
Output can be found at src/output_video/out_two_way.avi. Currently limited to first 20 frames.


## Credits
### YOLOv4 pytorch implementation by Tianxiaomo
<https://github.com/Tianxiaomo/pytorch-YOLOv4>

### BTS (State of the Art Monocular Depth Estimation)
- <https://paperswithcode.com/paper/from-big-to-small-multi-scale-local-planar>
- <https://github.com/Navhkrin/Bts-PyTorch>


