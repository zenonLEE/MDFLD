import cv2
from glob import glob
import os

mp4_dir = 'D:\이원명/연구과제/dmlab2023/video_me/'
mp4s = glob(mp4_dir+'/*.mp4')


for i in range(len(mp4s)):
    mp4_ = mp4s[i]
    filename = os.path.basename(mp4_)
    save_dir = os.path.join('video_me', filename.replace('.mp4', ''))
    os.makedirs(save_dir, exist_ok=True)

    mp4 = mp4s[i]
    videoCapture = cv2.VideoCapture(mp4)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    count = 0
    for j in range(int(fNUMS)):
        print(count)

        success, frame = videoCapture.read()
        if frame is not None:
            save = save_dir+'/'+str(count)+'.jpg'
            cv2.imwrite(save, frame)

        count+=1

    print(mp4, 'is finashed!')
    #videoCapture.release()
