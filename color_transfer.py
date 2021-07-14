import os
import cv2
import numpy as np

def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = x_mean.reshape(1,1,3) 
    x_std  = x_std.reshape(1,1,3) 
    return x_mean, x_std

def color_transfer(src, target):
    src    = src.astype(np.float32)
    target = target.astype(np.float32)
    
    s_mean, s_std = get_mean_and_std(src)
    t_mean, t_std = get_mean_and_std(target)

    height, width, channel = src.shape
    
    src = (src-s_mean)*(t_std/s_std) + t_mean
    src = np.clip(src, 0, 255)
    
    return src.astype(np.uint8)

targetfs = os.listdir('target/')
for file in targetfs:
    src = cv2.imread('Image018_3500_W.png')
    src = cv2.cvtColor(src,cv2.COLOR_BGR2LAB)

    tf = 'target/' + file
    tf = cv2.imread(tf)
    tf = cv2.cvtColor(tf,cv2.COLOR_BGR2LAB) 

    src = color_transfer(src,tf)

    src = cv2.cvtColor(src,cv2.COLOR_LAB2BGR)
    cv2.imwrite('result/' + file,src)