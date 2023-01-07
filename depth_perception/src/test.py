#!/usr/bin/env python3

import numpy as np
import cv2
 
# Load an color image in grayscale
img = cv2.imread('/home/altair/interbotix_ws/src/depth_perception/states/state.jpg')
res = np.float32(img)
res = res*(1/255.0)
 
# show image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
