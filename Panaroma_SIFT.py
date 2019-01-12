import cv2
import numpy as np
import sys
import os 
import random

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
kp1,dest1 = sift.detectAndCompute(gray1,None)
kp2,dest2 = sift.detectAndCompute(gray2,None)
'''
I was getting the same error. But in my case it was because I was using SIFT with cv2.NORM_HAMMING metric
in cv2.BFMatcher_create. Changing the metric to cv2.NORM_L1 solved the issue.
'''

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(dest1,dest2)
dmatches = sorted(matches, key = lambda x:x.distance)


src_pts  = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

# find homography matrix and do perspective transform
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
print("Homography Matrix is : \n")
print("----------------------------------------------------------------")
print(M)
res = cv2.drawMatches(img1, kp1, img2, kp2, dmatches[:10],None,flags=2)

h1,w1 = img1.shape[:2]
h2,w2= img2.shape[:2]
pts1 = np.float32([ [0,0],[0,h1],[w1-1,h1-1],[w1,0] ]).reshape(-1,1,2)
pts2 = np.float32([ [0,0],[0,h2],[w2-1,h2-1],[w2,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts2,M)
pts = np.concatenate((pts1,dst),axis=0)
[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
t = [-xmin,-ymin]
Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 
result = cv2.warpPerspective(img1, Ht.dot(M), (xmax-xmin, ymax-ymin))
result[t[1]:h1+t[1],t[0]:w1+t[0]] = img2


cv2.imwrite('task1_pano.jpg',result)

