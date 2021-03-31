# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:45:47 2019

@author: lior0
"""

import numpy as np
import cv2

img = cv2.imread('ofak1.jpg')
img0 = cv2.imread('ofak1.jpg',0)

#cv2.imwrite('img.jpg',img)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#clahe = cv2.createCLAHE()
cl1 = clahe.apply(img0)

#cv2.imwrite('clahe_1.jpg',cl1)


res = np.hstack((img0,cl1)) #stacking images side-by-side
cv2.imwrite('res.jpg',res)

#cv2.imshow('0',res)
#cv2.waitKey(50)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cl2 = hsv
cl2[:,:,0] = clahe.apply(hsv[:,:,0])
cl2 = cv2.cvtColor(cl2, cv2.COLOR_HSV2BGR)


#res2 = np.hstack((img,cl2)) #stacking images side-by-side
#cv2.imwrite('res2.jpg',res2)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cl2_2 = hsv
cl2_2[:,:,0] = clahe.apply(hsv[:,:,0])
cl2_2[:,:,1] = clahe.apply(hsv[:,:,1])
cl2_2[:,:,2] = clahe.apply(hsv[:,:,2])
cl2_2 = cv2.cvtColor(cl2_2, cv2.COLOR_HSV2BGR)


#res2_2 = np.hstack((img,cl2_2)) #stacking images side-by-side
#cv2.imwrite('res2_2.jpg',res2_2)


#cv2.imshow('0',res2)
#cv2.waitKey(50)


cl3 = img
cl3[:,:,0] = clahe.apply(img[:,:,0])
cl3[:,:,1] = clahe.apply(img[:,:,1])
cl3[:,:,2] = clahe.apply(img[:,:,2])

img = cv2.imread('ofak1.jpg')


#res3 = np.hstack((img,cl3)) #stacking images side-by-side
#cv2.imwrite('res3.jpg',res3)


#cv2.imshow('0',res3)
#cv2.waitKey(50)

res_all = np.hstack((img,cl2,cl2_2,cl3)) #stacking images side-by-side
cv2.imwrite('res_all.jpg',res_all)



