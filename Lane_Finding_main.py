# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:03:23 2017

@author: Emil Wåreus
"""

import cv2
import matplotlib.pylab as plt


import numpy as np



from Thresh import Thresh
from Distort import Distort
from Road_lanes import Lanes



Lanes = Lanes()

def process_img(image):
    
    #Due to that openCV is used to read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Undistort image
    image = cv2.undistort(image, mtx, dist, None, mtx)

    #Run the threashold algorithms, this Consists of: 
    #dir_binary = Thresh.dir_threashold(image, sobel_kernel=sobel_kernel, thresh=dir_thresh)
    #gradx = Thresh.abs_sobel_thresh(image, sobel_kernel=sobel_kernel, orient = 'x', thresh = x_thresh)
    #grady = Thresh.abs_sobel_thresh(image, sobel_kernel=sobel_kernel, orient = 'y', thresh = y_thresh)
    #mag = Thresh.mag_threashold(image, sobel_kernel=sobel_kernel, thresh=mag_thresh)
    #color = Thresh.color_thresh(image, thresh_s =(170, 255), thresh_h = (15, 100))
    #combined = np.zeros_like(dir_binary)
    #combined[(((gradx == 1) & (grady == 1) & (color == 1)) | (((mag == 1)| (color == 1)) & (dir_binary == 1)) )] = 1
                 
    combined = Thresh.threshold(image)

    #Perspective change
    per_img_C, M_C, src, M_t_c = Distort.perspect_Transform(combined)
    
    
    left_fitx, right_fitx, ploty, exist = Lanes.get_lane(per_img_C)
    
    
    img_with_lines = Lanes.draw_lines(image, per_img_C, left_fitx, right_fitx, ploty, M_t_c, exist)
       
    per_img_C= cv2.resize(per_img_C, None, fx = 0.5, fy= 0.5,  interpolation = cv2.INTER_CUBIC)
    combined = cv2.resize(combined, None, fx = 0.5, fy= 0.5,  interpolation = cv2.INTER_CUBIC)
   
    img = np.zeros([ 720,int(1280*1.5), 3], dtype=type(img_with_lines[0,0,0]))
    
    img[:,:1280,:] = img_with_lines
    print()
    img[:int(720/2),1280:,1] = (combined/combined.max())*255
    img[int(720/2):,1280:,0] = (per_img_C/per_img_C.max())*255
    
    img = cv2.resize(img, (1280, 720)) 
        
    return img,  left_fitx, right_fitx, ploty, exist, combined, per_img_C

def get_lane_image(image):
    '''
    This methode only returns the processed image.
    Note that it wants a BGR and returns a RGB
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_with_lines,  left_fitx, right_fitx, ploty, exist, combined, per_img_C = process_img(image) 
    
    #Remove this when makeing movie
    #img_with_lines = cv2.cvtColor(img_with_lines, cv2.COLOR_RGB2BGR)
    return img_with_lines



mtx, dist = Distort.calibrate()
for i in range(1,6):
   
    image = cv2.imread('test_images/test{}.jpg'.format(i)) 
    
    #Pipeline
    Lanes.init_lanes()
    img_with_lines, left_fitx, right_fitx, ploty, exist, combined, per_img_C = process_img(image)
    #Thresholds of image
    
    plt.figure(figsize= (24, 9))
    plt.imshow(img_with_lines) #, cmap='gray'

#Lets make some Movies: 
MakeMovie = False
if MakeMovie ==True:
    from moviepy.editor import VideoFileClip

    
    vid_output = 'output_1.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    vid_clip = clip1.fl_image(get_lane_image)
    vid_clip.write_videofile(vid_output, audio=False)
    
    
