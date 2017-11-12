# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 07:28:18 2017

@author: Emil Wåreus
"""


import numpy as np
import cv2
import matplotlib.pylab as plt
from Line import Line

class Lanes:
    
    
    def init_lanes(self):
        self.left_lane = Line()
        self.right_lane = Line()
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.left_fitx = [0] 
        self.right_fitx = [0]
        
         
    
    def find_curvature(self, yvals, fitx):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(yvals)
        # Define conversions in x and y from pixels space to meters
        
        #print(type(fitx), type(yvals))
        fit_cr = np.polyfit(yvals*self.ym_per_pix, fitx*self.xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return curverad
    
    
    def check_distance(self, left_fitx, right_fitx):
        bottom_leftx = left_fitx[-1]
        bottom_rightx = right_fitx[-1]
        if((bottom_rightx - bottom_leftx < 1000) & (bottom_rightx - bottom_leftx > 200)):
            self.left_lane.detected = True
            self.right_lane.detected = True
        else: 
            self.left_lane.detected = False
            self.right_lane.detected = False
    
    def sanity_check(self, lane, curverad, fitx, fit):       
        # Sanity check for the lane
        if lane.detected: # If lane is detected
            # If sanity check passes
            if abs(curverad / lane.radius_of_curvature - 1) < .6:        
                lane.detected = True
                lane.current_fit = fit
                lane.allx = fitx
                lane.bestx = np.mean(fitx)            
                lane.radius_of_curvature = curverad
                lane.current_fit = fit
            # If sanity check fails use the previous values
            else:
                lane.detected = False
                fitx = lane.allx
        else:
            # If lane was not detected and no curvature is defined
            if lane.radius_of_curvature: 
                if abs(curverad / lane.radius_of_curvature - 1) < 1:            
                    lane.detected = True
                    lane.current_fit = fit
                    lane.allx = fitx
                    lane.bestx = np.mean(fitx)            
                    lane.radius_of_curvature = curverad
                    lane.current_fit = fit
                else:
                    lane.detected = False
                    fitx = lane.allx      
            # If curvature was defined
            else:
                lane.detected = True
                lane.current_fit = fit
                lane.allx = fitx
                lane.bestx = np.mean(fitx)
                lane.radius_of_curvature = curverad
        return fitx
     
            
            
        
    def slid_window(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
     
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        #out_img = out_img.astype(np.int)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        plt.figure(3, figsize = (18,9))
        
        
        
        #if ((type(leftx) != None) & (type(lefty) != None) & (type(rightx) != None) & (type(righty) != None)): 
        if((leftx.size>0) & (lefty.size>0) & (rightx.size>0) & (righty.size>0)):
            #Fit a second order polynomial to each
            exist = True
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_lane.recent_xfitted = left_fit
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_lane.recent_xfitted = right_fit
            #print("left :",left_fit)
            #print("right :",right_fit)
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
                      
        else:
            
            

           
            left_fitx = [0]
            right_fitx = [0]
            ploty = ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            exist = False 
            
            
            
        
    
       
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        
        return left_fitx, right_fitx, ploty, exist
    
    def track_lanes(self, binary_warped):
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        margin = 100
        
        
        left_lane_inds = ((nonzerox > ( self.left_lane.recent_xfitted[0]*(nonzeroy**2) + self.left_lane.recent_xfitted[1]*nonzeroy +  
                                       self.left_lane.recent_xfitted[2] - margin)) & (nonzerox < ( self.left_lane.recent_xfitted[0]*(nonzeroy**2)+ 
                                       self.left_lane.recent_xfitted[1]*nonzeroy +  self.left_lane.recent_xfitted[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_lane.recent_xfitted[0]*(nonzeroy**2) + self.right_lane.recent_xfitted[1]*nonzeroy + 
                                        self.right_lane.recent_xfitted[2] - margin)) & (nonzerox < (self.right_lane.recent_xfitted[0]*(nonzeroy**2) + 
                                        self.right_lane.recent_xfitted[1]*nonzeroy + self.right_lane.recent_xfitted[2] + margin)))  
        
        #L and R pixel Pos
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
            
        if((leftx.size>0) & (lefty.size>0) & (rightx.size>0) & (righty.size>0)):
            
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_lane.recent_xfitted = left_fit
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_lane.recent_xfitted = right_fit
           
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            exist = True
          
            
        else:
            
            
            left_fitx = [0]
            right_fitx = [0]
            ploty = ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            exist = False 

        return left_fitx, right_fitx, ploty, exist
        
        
        
    
    def get_lane(self, binary_warped):
        
        #Check if distance between Lanes are ok, else slide window
        self.check_distance(self.left_fitx, self.right_fitx)
        
        
        if(self.right_lane.detected == False & self.right_lane.detected == False):
            
            self.left_fitx, self.right_fitx, ploty, exist = self.slid_window(binary_warped)
        else:
          
            self.left_fitx, self.right_fitx, ploty, exist = self.track_lanes(binary_warped)
          
        
    
        self.left_lane.radius_of_curvature  = self.find_curvature(ploty, self.left_fitx)
        self.right_lane.radius_of_curvature = self.find_curvature(ploty, self.right_fitx)
        
        # Sanity check for the lanes
        
        
        self.left_fitx  = self.sanity_check(self.left_lane, self.left_lane.radius_of_curvature, self.left_fitx, self.left_lane.recent_xfitted)
        self.right_fitx = self.sanity_check(self.right_lane, self.right_lane.radius_of_curvature, self.right_fitx, self.right_lane.recent_xfitted)

       
            
            
        
        return self.left_fitx, self.right_fitx, ploty, exist
        
        
    
    
    def draw_lines(self, image, binary_warped, left_fitx, right_fitx, ploty, M_t, exist=True):
        '''
        This function unwarps the lines and draws them to the image. 
        '''
        
        #Make wraper to draw lines on
        warp_zeros = np.zeros_like(binary_warped).astype(np.uint8)
        warp_wrap = np.dstack((warp_zeros,warp_zeros,warp_zeros))
        wrap = np.zeros_like(warp_wrap)
        
        if (exist==True):
            #Prepare for cv2.fillpoly
            left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            points = np.hstack((left,right))
            
            
            cv2.fillPoly(warp_wrap, np.int_([points]), (0, 0 , 255))
            
            #Warp wrap to original image
            wrap = cv2.warpPerspective(warp_wrap, M_t, (image.shape[1], image.shape[0]))
        
        
        wraped_image = cv2.addWeighted(image, 1, wrap, 0.3, 0)
        
        bottom_leftx = left_fitx[-1]
        bottom_rightx = right_fitx[-1]
        
        lane_center = (bottom_leftx + bottom_rightx) / 2
        
        car_center = 1280 / 2
        
        difference = lane_center - car_center
        
        self.left_lane.line_base_pos  = difference * self.xm_per_pix
        self.right_lane.line_base_pos= self.left_lane.line_base_pos
        if difference < 0:
            cv2.putText(wraped_image, "{0:.2f} m right of center".format(self.left_lane.line_base_pos), (30,110), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=6)
        else:
            cv2.putText(wraped_image, "{0:.2f} m left of center".format(self.left_lane.line_base_pos), (30,110), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=6)
        
        
        cv2.putText(wraped_image, "{0:.2f} m radius of curvature".format(float(self.left_lane.radius_of_curvature+self.right_lane.radius_of_curvature)/2), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=6)
        return wraped_image
    
        
 
    def visual_lines(self, binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzeroy, nonzerox):
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        plt.figure(3, figsize = (18,9))
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        