import os
import torch
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from monodepth.main_monodepth_pytorch import Model
import time


# Initiate model
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
model_test = Model(dict_parameters_test)
model_test.test()
disp = np.load('data/output/disparities_pp.npy')  # Or disparities.npy for output without post-processing
t1 = time.time()
print(t1-t0)



# Inference
print('=====================')
t0 = time.time()

disp, disp_pp, original_size = model_test.retest("image/image/init.png")
shape = disp.shape
shape2 = disp_pp.shape
print(shape, shape2, original_size)

t1 = time.time()

disp_to_img = skimage.transform.resize(disp.squeeze(), original_size, mode='constant')
print(disp_to_img.shape)
plt.imshow(disp_to_img, cmap='plasma')
plt.savefig('demo_depth.png')

print("retest:", t1-t0)


