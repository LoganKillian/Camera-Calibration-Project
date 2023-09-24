# References:
#https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html

import numpy as np
import cv2
import glob
import os

cap = cv2.VideoCapture(0)
i = True
a = str(2)


while(i == True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img_path = 'C:/Users/killi/Documents/Test Images'
    img = cv2.imread(img_path)
       
    # Our operations on the frame come here
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
       cv2.imwrite('image' + a + '.jpg', frame)
       a = int(a) + 1
       a = str(a)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        i = False


# Release the capture
cap.release()
cv2.destroyAllWindows()