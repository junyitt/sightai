import cv2
import argparse
import numpy as np
from sightai.video import SightVideo


def get_args():
    parser = argparse.ArgumentParser('Test on an image.')
    
    parser.add_argument('-input', type=str,
                        default='media/two_way.mp4',
                        help='path of your video file.', dest='vidpath')

    parser.add_argument('-fps', type=float,
                        default=30.0,
                        help='Input video fps.', dest='input_fps')

    parser.add_argument('-max', type=int,
                        default=20,
                        help='Maximum number of frames in output video.', dest='max_output_frames')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    S = SightVideo(vidpath = args.vidpath, input_fps = args.input_fps, max_output_frames = args.max_output_frames)
    