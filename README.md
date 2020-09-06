# sight.ai
An AI visual assistant made for visually impaired person to navigate around the city.

## Inspiration
In 2015, David Eagleman presented his remarkable research on TED by showing us that we can create new senses for ourselves. This can be done by taking an input data and translating them to vibrational mappings on the sensory vest. Then, a person can "feel" the input data through the vibration patterns on the sensory vest and learn from it. During the [TED presentation](https://www.ted.com/talks/david_eagleman_can_we_create_new_senses_for_humans?language=en), a deaf person was able to "hear" the words uttered to him via the sensory vest, and correctly write those words. 

This project aims to do the same on vision by creating a Vision Encoding System to aid a visually impaired person to "see" what is in front of him/her. This system will encode the image (seen from the perspective of a person) into depth map, which could help gauge distances between obstacles and people to the person seeing them. Then, the depth map is translated to vibrational mappings on the sensory vest to allow the visually impaired person to "feel" what is in front. 

## Architecture of Vision Encoding System
1. Image -> Infer Depth Map
2. Define 36 (arbitrary) grid area to represent each vibrational motor on the sensory vest.
3. Depth Map -> Calculate mean on Depth Map (for each grid area) to produce Vibrational Mappings.


## Getting Started
### Create conda environment
```
conda env create --file environment.yml
conda activate sightai
cd src
```

### Download Model Weights
Download weights for [BTS](https://drive.google.com/file/d/1_mENn0G9YlLAAr3N8DVDt4Hk2SBbo1pl/view) and [YOLOv4](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view) (see Credits). Place the weights in:
```
./pretrained/bts_latest
./pretrained/yolov4.weights
```

### Example scripts:
#### Run demo on sample image
```
python run_demo_image.py -input media/165_R.png -plot 1 -cuda 1
```
Output can be found at src/output/_165_R.png

#### Run demo on sample video
```
python run_demo_video.py -input media/two_way.mp4 -fps 30.0 -max 20 -cuda 1
```
Output can be found at src/output_video/out_two_way.avi. Currently limited to first 20 frames.
  
Note: Set -cuda 0 to run on cpu.

### Credits
#### YOLOv4 pytorch implementation by Tianxiaomo
<https://github.com/Tianxiaomo/pytorch-YOLOv4>

#### BTS (State of the Art Monocular Depth Estimation)
- <https://paperswithcode.com/paper/from-big-to-small-multi-scale-local-planar>
- <https://github.com/Navhkrin/Bts-PyTorch>


