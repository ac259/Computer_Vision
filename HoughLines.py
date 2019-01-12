import cv2
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import os

'''
You must have extracted the original_imgs.zip folder to the current working directory

'''
cwd = os.getcwd()
path = os.path.join(cwd,'original_imgs')
img = cv2.imread(os.path.join(path,'hough.jpg'),0)
#cv2.imshow("original_image",img)
X = img.shape[0]
Y = img.shape[1]
	
kernel_x=np.array((
[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]), dtype="int")

kernel_y= np.array((
[-1, -2, -1],
[0, 0, 0],
[1, 2, 1]), dtype="int")

H = kernel_x.shape[0]
W = kernel_y.shape[1]
#create a zero array
def Zero_array(output):
	output = [[0 for x in range(Y)] for y in range(X)]
	output = np.array(output)
	return output
#find max values
def max_value(matrix):
	max=0
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if(max<matrix[i][j]):
				max=matrix[i][j]
	return max
#eliminate zeros by second method
def final(matrix,image,max):
	for x in range(image.shape[1]):
		for y in range(image.shape[0]):
			matrix[y,x]=abs(image[y,x])/max
	return matrix
#normalize the image
def normalize(matrix_normalization):
	matrix_normalization=(matrix_normalization*255).astype("uint8")
	return matrix_normalization
#convolve the image
def convolve(img,kernel):
	out = []
	out = Zero_array(out)
	out = out.astype("double")
	for i in range(H, X-H):
		for j in range(W, Y-W):
			sum=0
			for m in range(H):
				for n in range(W):
					sum += kernel[m][n]*img[i-H+m][j-W+n]
			out[i,j] = sum
	return out

Gx = convolve(img,kernel_x)
#cv2.imshow("Sobel_x",Gx.astype("uint8"))
Gy = convolve(img,kernel_y)
#cv2.imshow("Sobel_y",Gy.astype("uint8"))
clean_x = []
clean_x = Zero_array(clean_x)
clean_x = clean_x.astype("double")

clean_y = []
clean_y = Zero_array(clean_y)
clean_y = clean_y.astype("double")

max1=max_value(Gx)
max1=max_value(Gy)

sobel_x = final(clean_x,Gx,max1)
sobel_y = final(clean_y,Gy,max1)

sobel_x=normalize(sobel_x)
sobel_y=normalize(sobel_y)

sobel=sobel_x+sobel_y
#--------------------------------------------------------------


def threshold(thres,image):
	img1=np.zeros_like(image)
	for i in range(X):
		for j in range(Y):
			if(image[i][j]>thres):
				img1[i][j]=255
			else:
				img1[i][j]=0
	return img1

image1=threshold(21,sobel_y)
image2=threshold(80,sobel_x)

def Hough_lines(image):
	theta = np.deg2rad(np.arange(-90.0, 90.0))
	#print('the',thetas)
	diag_len = int(round(math.sqrt(X*X+Y*Y)))
	#print('dia',diag_len)  # max_dist
	rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
	cos_t = np.cos(theta)
	sin_t = np.sin(theta)
	num_theta = len(theta)
	accumulator = np.zeros((2 * diag_len, num_theta),dtype=np.uint8)
	y,x= np.nonzero(image)
	for i in range(len(x)):
	    x_id = x[i]
	    y_id= y[i]
	    for t in range(num_theta):
	    	rho = int(round(x_id * cos_t[t] + y_id * sin_t[t])) + diag_len
	    	accumulator[rho, t] += 1

	return accumulator,theta,rhos

def hough_peaks(accumulator, no_of_lines):
    peak_list = []
    tmax = 0
    temp_coord = []
    for i in range(no_of_lines):
        for x in range(accumulator.shape[0]):
            for y in range(accumulator.shape[1]):
                if accumulator[x, y] > tmax:
                    t = [x, y]
                    if t not in peak_list:
                        tmax = accumulator[x, y]
                        temp_coord = t
        peak_list.append(temp_coord)
        temp_coord = []
        tmax = 0
    return peak_list

l,w = image1.shape

def drawing_lines(image,color,flag):
	acc,t,r=Hough_lines(image)
	#cv2.imwrite("accumulator.jpg",acc)
	peaks = hough_peaks(acc,8)
	#print(peaks)

	output_image = cv2.imread(os.path.join(path,"hough.jpg"))

	for peak in peaks:
		print(peak)
		x,y = peak
		#print(x,y)
		rho1 = r[x]
		theta1 = t[y]
		
		y1= int((rho1 - 0 * math.cos(theta1)) / math.sin(theta1) )
		#print(y1)
		y2 = int((rho1 - w * math.cos(theta1)) / math.sin(theta1)) 
		#print(y2)
		#color = (0,0,255)
		cv2.line(output_image,(0,y1),(w,y2),color,3)

	#cv2.imshow("output",output_image)
	if flag == 'r':
		cv2.imwrite('red_lines.jpg',output_image)
	else:
		cv2.imwrite('blue_lines.jpg',output_image)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()


drawing_lines(image2,(0,0,255),'r')
drawing_lines(image1,(255,0,0),'b')