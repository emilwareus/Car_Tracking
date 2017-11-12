# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 07:17:24 2017

@author: Emil Wåreus
"""

import cv2
import Lane_Finding_main


cap = cv2.VideoCapture('challenge_video.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if type(frame) != None:
        frame = cv2.resize(frame, (1280, 720)) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_with_lines = Lane_Finding_main.get_lane_image(frame)
        #print(frame.shape)
        #Display the resulting frame
        img_with_lines = img_with_lines
        print(img_with_lines.max())
        cv2.imshow('frame', img_with_lines)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
 

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
