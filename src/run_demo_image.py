import argparse
from sightai.module import SightAI


def get_args():
    parser = argparse.ArgumentParser('Test on an image.')
    parser.add_argument('-plot', type=int,
                        default=1,
                        help='Output depth map.', dest='plot')
    
    parser.add_argument('-input', type=str,
                        default='media/165_R.png',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    S = SightAI(use_cuda = True)
    S.inference(args.imgfile, plot = args.plot)
    # S.inference("media/001_L.png", plot = True)
    # S.inference("media/165_R.png", plot = False)