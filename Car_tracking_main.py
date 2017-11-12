# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:19:20 2017

@author: Emil Wåreus
"""

import cv2
import matplotlib.pylab as plt


import numpy as np



from Thresh import Thresh
from Distort import Distort
from Road_lanes import Lanes

for i in range(1,6):
   
    image = cv2.imread('test_images/test{}.jpg'.format(i)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
    plt.figure(figsize= (24, 9))
    plt.imshow(image) #, cmap='gray'
    
         
              
              
              
MakeMovie = False
if MakeMovie ==True:
    from moviepy.editor import VideoFileClip

    
    #vid_output = 'output_1.mp4'
    #clip1 = VideoFileClip("project_video.mp4")
    #vid_clip = clip1.fl_image(get_lane_image)
    #vid_clip.write_videofile(vid_output, audio=False)
    
