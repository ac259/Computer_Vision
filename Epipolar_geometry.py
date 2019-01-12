import cv2
import numpy as np
import sys
import os 
import random
import matplotlib.pyplot as plt
'''
You must have extracted the proj2_data.zip folder to the current working directory

'''
cwd = os.getcwd()
path = os.path.join(cwd,'data')
img1 = cv2.imread(os.path.join(path,'tsucuba_left.png'))
img2 = cv2.imread(os.path.join(path,'tsucuba_right.png'))

gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
r,c = gray1.shape

sift = cv2.xfeatures2d.SIFT_create()
kp1,dest1 = sift.detectAndCompute(gray1,None)
kp2,dest2 = sift.detectAndCompute(gray2,None)

cv2.imwrite('task2_sift1.jpg',cv2.drawKeypoints(gray1,kp1,img1))
cv2.imwrite('task2_sift2.jpg',cv2.drawKeypoints(gray2,kp2,img2))

bf=cv2.BFMatcher()
matches = bf.knnMatch(dest1,dest2,k=2)
good = [] 
pts1 = []
pts2 = []
for m,n in matches:
	if m.distance<0.75*n.distance:
		good.append(m)
image3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)
cv2.imwrite('task2_matches_knn.jpg',image3)

pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ])
pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ])


F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
print("Fundamental Matrix is: \n")
print("----------------------------------------------------------------")
print(F)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(image1,image2,lines,pts1,pts2):
    img1 = image1
    img2 = image2
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

img1 = cv2.imread(os.path.join(path,'tsucuba_left.png'))
img2 = cv2.imread(os.path.join(path,'tsucuba_right.png'))

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=23)
disparity = stereo.compute(gray1,gray2)


plt.imsave('task2_disparity.png',disparity,cmap ='gray')

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)

lines1_10 = lines1
random.shuffle(lines1_10)
lines1_10 = lines1_10[0:20]
print(len(lines1_10))
img5,img6 = drawlines(img1,img2,lines1_10,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
lines2_10 = lines1
random.shuffle(lines2_10)
lines2_10 = lines2_10[0:20]
print(len(lines2_10))
img3,img4 = drawlines(img2,img1,lines2_10,pts2,pts1)

cv2.imwrite('task2_epi_left.jpg',img5)
cv2.imwrite('task2_epi_right.jpg',img3)
