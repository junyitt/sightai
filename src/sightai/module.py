import os
import cv2
from PIL import Image
import time
import skimage.transform
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from logzero import setup_logger, logging 
from monodepth.main_monodepth_pytorch import Model as DepthModel
from tool.darknet2pytorch import Darknet
from tool.utils import plot_boxes_cv2, load_class_names
from tool.torch_utils import do_detect

class SightAI:
    def __init__(self, use_cuda = True):
        self.use_cuda = use_cuda
        self.init_log()
        self.init_depth_model()
        self.init_yolo_model()
    
    def init_log(self):
        self.log = setup_logger(name="sightailog", logfile="./logs/sightai.txt", level=logging.INFO)
        self.log.info("1 - Initiated log.")

    def init_yolo_model(self):
        t0 = time.time()
        cfgfile = "./cfg/yolov4-tiny.cfg"
        cfgfile = "./cfg/yolov4.cfg"
        weightfile = "./pretrained/yolov4.weights"
        namesfile = 'data/coco.names'
        use_cuda = self.use_cuda

        self.darknet_model = Darknet(cfgfile)
        self.yolo_width, self.yolo_height = (self.darknet_model.width, self.darknet_model.height)
        self.darknet_model.load_weights(weightfile)
        if use_cuda:
            self.darknet_model.cuda()
        self.class_names = load_class_names(namesfile)

        t1 = time.time()
        total_time = round(t1-t0, 2)
        self.log.info("1 - Initiated YOLOv4. -- {} minutes {} seconds".format(total_time//60, total_time % 60))

    def init_depth_model(self):
        t0 = time.time()
        dict_parameters_test = edict({'data_dir':'./image',
                                    'model_path':'./pretrained/monodepth_resnet18_001.pth',
                                    'output_directory':'data/output/',
                                    'input_height':256,
                                    'input_width':512,
                                    'model':'resnet18_md',
                                    'pretrained':False,
                                    'mode':'test',
                                    'device':'cuda:0',
                                    'input_channels':3,
                                    'num_workers':0,
                                    'use_multiple_gpu':False})
        self.depth_model = DepthModel(dict_parameters_test)
        self.depth_model.test()
        t1 = time.time()
        total_time = round(t1-t0, 2)
        self.log.info("1 - Initiated DepthModel. -- {} minutes {} seconds".format(total_time//60, total_time % 60))

    def bbox_inference(self, img_path):
        img = cv2.imread(img_path)
        sized = cv2.resize(img, (self.yolo_width, self.yolo_height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        boxes = do_detect(self.darknet_model, sized, 0.4, 0.6, self.use_cuda)
        return img, boxes

    def inference(self, img_path, plot = False):
        t0 = time.time()
        
        # Depth estimation
        disp, disp_pp, original_size = self.depth_model.retest(img_path)
        
        # Bounding box
        img, boxes = self.bbox_inference(img_path)

        # Plot
        if plot:
            plt.imshow(disp_pp, cmap='plasma')
            plt.savefig('demo_depth.png')

            plt.imshow(disp, cmap='plasma')
            plt.savefig('demo_depth2.png')

            plot_boxes_cv2(img, boxes[0], savename='demo_yolo.jpg', class_names=self.class_names)

        t1 = time.time()
        total_time = round(t1-t0, 2)
        self.log.info("1 - Done inference_image. -- {} minutes {} seconds".format(total_time//60, total_time % 60))

