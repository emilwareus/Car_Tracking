# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 07:29:41 2017

@author: Emil Wåreus
"""


import numpy as np
import cv2
import matplotlib.pylab as plt
import matplotlib.image as mpimg


class Distort:
    
    def calibrate():
        
        nx = 9
        ny = 6
        
        #Getting points ready
        pnt = np.zeros((nx*ny,3), np.float32)
        pnt[:,:2] =  np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        
        obj_pnt = []
        img_pnt = []
        
        
        for i in range(1,21):
            
                
            cal_img = ('camera_cal/calibration{}.jpg'.format(i))
            img = cv2.imread(cal_img)    
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            if ret == True:
                
                obj_pnt.append(pnt)
                img_pnt.append(corners)
                
            
        img_size = (img.shape[1], img.shape[0])    
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pnt, img_pnt, img_size,None,None)
                
        for i in range(1,21):
            
            cal_img = ('camera_cal/calibration{}.jpg'.format(i))
            img = cv2.imread(cal_img)    
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            if ret == True:
                
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                
                dst = cv2.undistort(img, mtx, dist, None, mtx)
       
                f1, (ax1, ax2) = plt.subplots(1, 2, figsize= (14, 5))
                f1.tight_layout()
                ax1.imshow(img)
                ax1.set_title('Raw image {}'.format(i), fontsize = 30)
                
                ax2.imshow(dst, cmap = 'gray')
                ax2.set_title('Distorted Image {}'.format(i), fontsize = 30)
                
                plt.subplots_adjust(left=0., right = 1, top = 0.9, bottom = 0.)
        
        return mtx, dist
    
    
    
    def unwarp(image, M_t):
        img_size = (image.shape[0], image.shape[1])
        
        unwarped = cv2.warpPerspective(image, M_t, img_size, flags = cv2.INTER_LINEAR)
        
        return unwarped        
        
    def perspect_Transform(img, square = [90,  450, 600, 650]):
        '''
        img is the image you want to transforme
        square contaions 4 variables
        [0] wb = 90
        [1] hb = 450
        [2] wt = 600
        [3] ht = 650
        '''
        H, W = (img.shape[0],img.shape[1])
        wb = square[0]
        hb = square[1]
        wt = square[2]
        ht = square[3]
        
        src = np.float32([[(W/2-wb), hb],[(W/2+wb), hb], [(W/2+wt), ht], [(W/2-wt), ht]])

        img_size = (img.shape[1], img.shape[0])
        offset = 0 #100
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                         [img_size[0]-offset, img_size[1]-offset], 
                                         [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        M_t = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, img_size)
        
        return warped, M, src, M_t

    