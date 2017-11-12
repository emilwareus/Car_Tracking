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
