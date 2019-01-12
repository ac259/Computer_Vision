import cv2
import numpy as np
import sys
import os 

'''
You must have extracted the proj2_data.zip folder to the current working directory

'''
cwd = os.getcwd()
path = os.path.join(cwd,'data')
img1 = cv2.imread(os.path.join(path,'mountain1.jpg'))
img2 = cv2.imread(os.path.join(path,'mountain2.jpg'))

gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()
kp1 = sift.detect(gray1,None)
kp2 = sift.detect(gray2,None)


cv2.imwrite('task1_sift1.jpg',cv2.drawKeypoints(gray1,kp1,img1))
cv2.imwrite('task1_sift2.jpg',cv2.drawKeypoints(gray2,kp2,img2))
