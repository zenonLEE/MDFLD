import numpy as np
import csv
import os

import matplotlib.pyplot as plt
from glob import glob
from PIL import Image, ImageDraw
import json

import torch
import cv2
import random
import math


def plot_verts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image,(int(st[0]), int(st[1])), 2, c, -1)

    return image

def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))

def load_lmd(lmd_dir):
    if lmd_dir.split('.')[-1] == 'pts':
        lmd = read_pts(lmd_dir)
    else:
        f = open(lmd_dir)
        lmd = json.load(f)
        lmd = np.array(lmd['landmarks']['points'])[:, ::-1]
    return lmd

def cal_scale_center(landmark):
    x1 = np.min(landmark[:, 0]);
    x2 = np.max(landmark[:, 0])
    y1 = np.min(landmark[:, 1]);
    y2 = np.max(landmark[:, 1])

    center_w = (math.floor(x1) + math.ceil(x2)) / 2.0
    center_h = (math.floor(y1) + math.ceil(y2)) / 2.0

    scale = max(math.ceil(x2) - math.floor(x1), math.ceil(y2) - math.floor(y1)) / 200.0
    return center_w, center_h, scale

def lmd68to5(landmark):
    point0 = (landmark[36, None,  :]+landmark[39,None,  :])/2
    point1 = (landmark[42, None, :]+landmark[45,None,  :])/2
    point2 = (landmark[33, None, :]+landmark[35, None, :])/2
    point3 = landmark[48,None,  :]
    point4 = landmark[54, None, :]
    return np.concatenate([point0, point1, point2, point3, point4], 0)

#af_data_dir = '/home/user/data/AF_dataset/AF_dataset/'
af_data_dir = './dataset/3dshop/'
img_dir_list = sorted(glob(af_data_dir+'/*/*.jpg'))
pts_dir_list = sorted(glob(af_data_dir+'/*/*.ljson'))

main_dir = './dataset/meta/3dshop/'
os.makedirs(main_dir, exist_ok=True)

with open(main_dir + 'test.tsv', 'w') as record_file:
    writer = csv.writer(record_file)
    writer.writerow(['',
                     # str(list(lmd5.flatten()))[1:-1],
                     # str(list(lmd.flatten()))[1:-1],
                     # scale,
                     # center_w,
                     # center_h
                     ])
    for i in range(len(img_dir_list)):
        print(img_dir_list[i])
       # print(pts_dir_list[i])
       # img_path = img_dir_list[i].replace('./dataset/AF_dataset_0.3/', './results')
        img_path = img_dir_list[i]
        #lmd = load_lmd(pts_dir_list[i])

        #lmd5 = lmd68to5(lmd)
        #center_w, center_h, scale = cal_scale_center(lmd)

        #img = cv2.imread(img_dir_list[i], cv2.IMREAD_COLOR)
        #plot_image = plot_verts(img, lmd, color='r')
        #save_dir = main_dir + img_dir_list[i].split('/')[-1]
        #cv2.imwrite(save_dir, plot_image)
        writer.writerow([img_path
                         # str(list(lmd5.flatten()))[1:-1],
                         # str(list(lmd.flatten()))[1:-1],
                         # scale,
                         # center_w,
                         # center_h
                         ])
        print()