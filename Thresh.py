# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 07:49:04 2017

@author: Emil Wåreus
"""

import numpy as np
import cv2
class Thresh: 
    def dir_threashold(img, sobel_kernel = 3, thresh = (0, np.pi/2)):
        
        #1 Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        #2 Gradient in X and Y
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
        #3 Calc dir of gradient
        dir_mask = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
        
        
        
        #Create mask
        output = np.zeros_like(dir_mask)
        output[(dir_mask >= thresh[0]) & (dir_mask <= thresh[1])] = 1
        
        return output
    
    def mag_threashold(img, sobel_kernel = 3, thresh = (0, 255)):
        
        #1 Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        #2 Gradient in X and Y
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
        #Magnitude
        mag = np.sqrt(sobelx**2 + sobely**2)
        
        #Scale
        scaled_sobel = np.uint8(255*mag/np.max(mag))
        
        
        #Create mask
        output = np.zeros_like(scaled_sobel)
        output[(scaled_sobel>= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        
        return output
    
    
    
    def abs_sobel_thresh(img, sobel_kernel=3, orient = 'x', thresh = (0, 255)):
        #1 Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        #2 Gradient in X and Y
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
            
        #Absolute
        abs_sobel = np.absolute(sobel)
        
        #Scale
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        
        
        #mask
        mask = np.zeros_like(scaled_sobel)
        mask[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        
    
        return mask
    
    def color_thresh(img, thresh_s = (70, 185), thresh_h = (15, 30)):
        
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        s_channel = hls[:,:,2]
        s_channel = (s_channel/(s_channel.max()))*255
        h_channel = hls[:,:,0]
        h_channel = (h_channel/(h_channel.max()))*255
        
        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 1
                 
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= thresh_h[0]) & (h_channel <= thresh_h[1])] = 1
        
       
       
        ##c_binary = np.dstack(s_binary, h_binary)*255
        c_binary = np.zeros_like(h_binary)
        c_binary[(h_binary == 1) & (s_binary == 1)] = 1
                    
        return c_binary
    
    
    def threshold(image, sobel_kernel = 3, dir_thresh = (.65, 1.05), mag_thresh = (60, 255), 
                  x_thresh = (10, 255), y_thresh = (60,255)):
        
        
        dir_binary = Thresh.dir_threashold(image, sobel_kernel=sobel_kernel, thresh=dir_thresh)
        gradx = Thresh.abs_sobel_thresh(image, sobel_kernel=15, orient = 'x', thresh = x_thresh)
        grady = Thresh.abs_sobel_thresh(image, sobel_kernel=15, orient = 'y', thresh = y_thresh)
        mag = Thresh.mag_threashold(image, sobel_kernel=sobel_kernel, thresh=mag_thresh)
        color = Thresh.color_thresh(image, thresh_s =(130, 255), thresh_h = (0, 130))
        
        
        combined = np.zeros_like(dir_binary)
       
        combined[(((gradx == 1) & (grady == 1)) |((mag == 1) & (dir_binary == 1)) | (color==1))] = 1
        #combined[(dir_binary==1)]=1
        # Defining vertices for marked area
        img_shape = image.shape
        l_b = (190, img_shape[0])
        r_b = (img_shape[1]-190, img_shape[0])
        apex1 = (int(img_shape[1]/2) - 60, 410)
        apex2 = (int(img_shape[1]/2) + 60, 410)
        inner_left_bottom = (150+190, img_shape[0])
        inner_right_bottom = (img_shape[1]-150-190, img_shape[0])
        inner_apex1 = (int(img_shape[1]/2) + 10,480)
        inner_apex2 = (int(img_shape[1]/2) - 10,480)
        vertices = np.array([[l_b, apex1, apex2, \
                              r_b, inner_right_bottom, \
                              inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)
        # Masked area
        mask = np.zeros_like(combined)   
        
        if len(img_shape) > 2:
            channel_count = img_shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        combined= cv2.bitwise_and(combined, mask)
                   
        return combined