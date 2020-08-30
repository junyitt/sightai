import os
import torch
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from monodepth.main_monodepth_pytorch import Model
import time

t0 = time.time()
dict_parameters_test = edict({'data_dir':'./image',
                              'model_path':'./monodepth_resnet18_001.pth',
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

print('=====================')

t0 = time.time()

disp, disp_pp, original_size = model_test.retest("image/image/001_L.png")
shape = disp.shape
shape2 = disp_pp.shape
print(shape, shape2, original_size)

t1 = time.time()

disp_to_img = skimage.transform.resize(disp.squeeze(), original_size, mode='constant')
print(disp_to_img.shape)
plt.imshow(disp_to_img, cmap='plasma')
plt.savefig('test.png')

print("retest:", t1-t0)



t0 = time.time()

disp, disp_pp, original_size = model_test.retest("image2/image/059_L.png")
shape = disp.shape
shape2 = disp_pp.shape

t1 = time.time()

disp_to_img = skimage.transform.resize(disp.squeeze(), original_size, mode='constant')
print(disp_to_img.shape)
plt.imshow(disp_to_img, cmap='plasma')
plt.savefig('test2.png')

print("retest2:", t1-t0)

print('=====================')

# t0 = time.time()

# disp, disp_pp = model_test.retest("image2/image/059_L.png")
# shape = disp.shape
# print(disp.shape)

# disp_to_img = skimage.transform.resize(disp[0].squeeze(), shape[1:3], mode='constant')
# plt.imshow(disp_to_img, cmap='plasma')
# plt.savefig('test2.png')

# t1 = time.time()
# print("retest:", t1-t0)
