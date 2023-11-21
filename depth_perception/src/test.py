#!/usr/bin/env python3

import numpy as np
import cv2
 
# Load an color image in grayscale
img1 = cv2.imread('/home/altair/interbotix_ws/src/depth_perception/states/state_45.jpg')
img2 = cv2.imread('/home/altair/interbotix_ws/src/depth_perception/states/state_46.jpg')
subtracted = cv2.subtract(img2, img1) #right order
#res = np.float32(img)
#res = res*(1/255.0)


# show image
cv2.imshow('image',subtracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
