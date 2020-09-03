import btsdepth.BTS as BTS
import torch
import cv2
import numpy as np
from DepthVisualizer import DepthRenderer
import argparse
import sys
import os
import albumentations as A
import skimage
import matplotlib.pyplot as plt

def crop_img(img):
    height, width, channels = img.shape
    top, left = int(height - 352), int((width - 1216) / 2)
    return img[top:top+352, left:left+1216]

def resize(img):
    img = cv2.resize(img, (1216,352))
    return img

def load_data_bts(img_path):
    img = cv2.imread(img_path)
    original_size = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.uint8)
    crop_img = resize(img)
    data = {'image': crop_img, 'label': crop_img}
    test_image_only_transforms = A.Compose([
        A.Normalize(always_apply=True)
    ])
    data = test_image_only_transforms(**data)
    data["image"] = torch.tensor(data["image"]).transpose(0, 2).transpose(1, 2)
    data["label"] = torch.tensor(data["label"]).transpose(0, 2).transpose(1, 2)
    return data, original_size

def predict_bts(model, data, original_size):
    res = original_size[0], original_size[1]
    result_raw = model.predict(data["image"], 715.0873)
    result_raw_resized = skimage.transform.resize(result_raw.squeeze(), res, mode='constant')
    mn, mx = np.min(result_raw_resized), np.max(result_raw_resized)
    rr = 255*(result_raw_resized-mn)/(mx-mn)
    rr = rr.astype(int)
    return rr

