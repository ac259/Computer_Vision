import numpy as np
import cv2
import sys
import os
def create_zero_array(name_of_array):
name_of_array = [[0 for x in range(image_height)] for y in range(image_width )]
name_of_array = np.array(name_of_array)
return name_of_array
# Convolve function - takes the kernel as an argument
def Convolve(kernel,output):
for x in range(image.shape[1]):
for y in range(image.shape[0]):
output[y,x] = (kernel*image_with_padding[y:y+3,x:x+3]).sum()
return output
# Finding the max pixel value
def findmax(matrix):
max1 = 0
for i in range(matrix.shape[0]):
for j in range(matrix.shape[1]):
if(max1 < matrix[i,j]):
max1 = matrix[i,j]
return max1
def cleaning_up_image(clean_image_matrix,image,max_value):
for x in range(image.shape[1]):
for y in range(image.shape[0]):
clean_image_matrix[y,x] = abs(image[y,x])/max_value
#print(clean_sobel_x[y,x])
return clean_image_matrix
def normalize(matrix_to_be_normalized):
matrix_to_be_normalized = (matrix_to_be_normalized * 255).astype("uint8")
return matrix_to_be_normalized
#Sobel x kernel
sobelX = np.array((
[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]), dtype="int")
#Sobel y kernel
sobelY = np.array((
[-1, -2, -1],
[0, 0, 0],
[1, 2, 1]), dtype="int")
# Read the image
image = cv2.imread(sys.argv[1],0)
cv2.imshow("Original Image",image)
image_height = image.shape[1]
image_width = image.shape[0]
# Padding the image
image_with_padding = [[0 for x in range(image_height +2)] for y in range(image_width +2)]
image_with_padding = np.array(image_with_padding)
clean_sobel_x = []
clean_sobel_x = create_zero_array(clean_sobel_x)
clean_sobel_x = clean_sobel_x.astype('double')
clean_sobel_y = []
clean_sobel_y = create_zero_array(clean_sobel_y)
clean_sobel_y = clean_sobel_y.astype('double')
# Padding Size- we need one pixel padding hence x and y are 1
x=1
y=1
image_with_padding[x:image.shape[0]+x, y:image.shape[1]+y] = image
sobel_y = []
sobel_y = create_zero_array(sobel_y)
sobel_y = Convolve(sobelY,sobel_y)
sobel_x = []
sobel_x = create_zero_array(sobel_x)
sobel_x = Convolve(sobelX,sobel_x)
cv2.imshow("Sobelx output",sobel_x.astype("uint8"))
cv2.imshow("Sobely output",sobel_y.astype("uint8"))
max1 = findmax(sobel_x)
max2 = findmax(sobel_y)
clean_sobel_x = cleaning_up_image(clean_sobel_x,sobel_x,max1)
clean_sobel_y = cleaning_up_image(clean_sobel_y,sobel_y,max2)
clean_sobel_x = normalize(clean_sobel_x)
clean_sobel_y = normalize(clean_sobel_y)
cv2.imshow("clean_sobel_x",clean_sobel_x)
cv2.imshow("clean_sobel_y",clean_sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()