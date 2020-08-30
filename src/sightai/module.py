import os
import cv2
import numpy as np
import pandas as pd
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


def construct_object_table(img, boxes, class_names, disp=None):
    boxes = boxes[0]
    width = img.shape[1]
    height = img.shape[0]

    d_list = []
    for i in range(len(boxes)):
        d = {}
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        cls_id = box[6]
        object_name = class_names[cls_id]
        conf = box[5]

        disp_values = []
        for x in range(x1, x2, 1):
            for y in range(y1, y2, 1):
                if x < disp.shape[0] and x > 0 and y > 0 and y < disp.shape[1]:
                    dp = disp[x,y]
                    disp_values.append(dp)

        xc, yc = (x1+x2)/2, (y1+y2)/2
        inzone = False
        if xc > width/3 and xc < 2*width/3 and yc < height and yc > height/2:
            inzone = True

        if xc < width/3:
            direction = "left"
        elif xc > 2*width/3:
            direction = "right"
        else:
            direction = "center"

        d["x1"] = x1
        d["y1"] = y1
        d["x2"] = x2
        d["y2"] = y2
        d["xc"] = xc
        d["yc"] = yc
        d["object_name"] = object_name
        d["inzone"] = inzone
        d["direction"] = direction
        # d["disp_values"] = disp_values
        dist = round(100*np.mean(disp_values),2)
        d["distance"] = dist
        # if object_name == "car":
        #     plt.hist(disp_values)
        #     plt.savefig("hist{}.png".format(i))
        d_list.append(d)

    # add "obstacle"
    disp_values = []
    w1, w2 = int(width/3), int(2*width/3)
    h1, h2 = int(height/2), int(height)
    for x in range(w1, w2, 1):
        for y in range(h1, h2, 1):
            dp = disp[x,y]
            disp_values.append(dp)
    dist = round(100*np.mean(disp_values),2)
    add_d = {"object_name": "obstacle", "inzone": True, "direction": "center", "distance": dist}
    d_list.append(add_d)
    df = pd.DataFrame(d_list).sort_values("distance")
    return df


def get_instructions(df):
    message = []
    u1 = df["inzone"]
    u2 = df["distance"] < 1
    df1 = df.loc[u1 & u2]
    if df1.shape[0] == 0:
        message.append("Continue straight.")
        return message

    if df1.shape[0] > 0:
        object_name = df1.iloc[0]["object_name"]
        direction = df1.iloc[0]["direction"]
        if direction == "center":
            message.append("{} in front.".format(object_name))
        else:
            message.append("{} on your {}.".format(object_name, direction))
    
    if df1.shape[0] > 1:
        direction = df1.iloc[1]["direction"]
        if direction == "left":
            message.append("Navigate right.")
        else:
            message.append("Navigate left.")
    else:
        # default left
        message.append("Navigate left.")
    return message

   
class SightAI:
    def __init__(self, use_cuda = True):
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.init_log()
        self.init_depth_model()
        self.init_yolo_model()
        self.inference(img_path = "image/image/init.png")
    
    def init_log(self):
        self.log = setup_logger(name="sightailog", logfile="./logs/sightai.txt", level=logging.INFO)
        self.log.info("1 - Initiated log.")

    def init_yolo_model(self):
        t0 = time.time()
        # cfgfile = "./cfg/yolov4-tiny.cfg"
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
                                    'device':'{}:0'.format(self.device),
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

    def inference(self, img_path, plot = False, j = 0):
        t0 = time.time()
        
        # Depth estimation
        disp, disp_pp, original_size = self.depth_model.retest(img_path)
        
        # Bounding box
        img, boxes = self.bbox_inference(img_path)    
        
        df = construct_object_table(img, boxes, self.class_names, disp)
        msg = get_instructions(df)

        # Plot
        if plot:
            # plt.imshow(disp_pp, cmap='plasma')
            # plt.savefig('demo_depth.png')

            plt.imshow(disp, cmap='plasma')
            plt.savefig('frame/depth_map{}.png'.format(j))

            plot_boxes_cv2(img, boxes[0], savename='frame/bbox{}.png'.format(j), class_names=self.class_names, disp = disp)

        t1 = time.time()
        total_time = round(t1-t0, 2)
        self.log.info("1 - Done inference_image {}. -- {} minutes {} seconds".format(img_path, total_time//60, total_time % 60))

        return msg, df
