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

bf = cv2.BFMatcher()
matches = bf.knnMatch(dest1,dest2,k=2)

# Apply ratio test
good = []
good_without_list = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        good_without_list.append(m)
#print("total number of matches",len(good))
# Good contains set of all matches - We need only 10 random images 
# shuffle the list and pick first 10

good_shuffle = good
random.shuffle(good_shuffle)
good_10 = good_shuffle[0:10]

# cv.drawMatchesKnn expects list of lists as matches.
# TypeError: Required argument 'outImg' (pos 6) not found
# For default img3 = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good,flags=2)
# We need to pass None instead of output image

img3_10 = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good_10,None,flags=2)
img3 = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good,None,flags=2)

img4 = cv2.drawMatches(gray1,kp1,gray2,kp2,good_without_list,None,flags =2)

cv2.imwrite('task1_matches_knn_10_random_matches.jpg',img3_10)
cv2.imwrite('task1_matches_knn.jpg',img3)
cv2.imwrite('task1_matches_knn_drawMatches.jpg',img4)
