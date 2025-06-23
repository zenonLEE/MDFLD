import cv2
from glob import glob
import os
from natsort import natsorted

file_name = 'video_me_pixar_d_1_80'
frame_dir = './pretrained_model/video_me/{}/image_results/'.format(file_name)
frames = natsorted(glob(frame_dir+'*.jpg'))

frameSize = (256, 256)
out = cv2.VideoWriter('./video_me/lmd_{}.mp4'.format(file_name),cv2.VideoWriter_fourcc('m', 'p', '4', 'v') , 20, frameSize)


for i in range(len(frames)):
    img = cv2.imread(frames[i])
    out.write(img)

    #videoCapture.release()

out.release()
