import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def point_detection(image):
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	x_cord = 0
	y_cord = 0
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
	mult  = []
	for y in np.arange(pad, iH - pad):
		for x in np.arange(pad, iW - pad):
			
			roi = image[y - pad:y - pad + 3, x - pad:x - pad + 3]
			sum = np.multiply(roi,kernel).sum()
			#k = abs((roi * kernel).sum())
			mult.append(sum)
			
			if(sum > 8382): # Calculated this value by taking 0.9 * the max(mult)
				output[y - pad, x - pad] = 1
				y_cord = y - pad
				x_cord = x - pad 
	return output

def segmentation(image):
	H=image.shape[0]
	W=image.shape[1]
	Threshold=203
	# We see from the histogram that the peak value is 192 - but when we 
	# segment the image we see that there is still some noise.
	# Through trial and error we see that the value of 204 renders a good result.
	img=np.zeros_like(image)

	[u,indices]=np.unique(image,return_counts=True)
	plt.plot(u[1:len(u)],indices[1:len(indices)])
	plt.savefig('output-histogram1')
	#print(np.argmax([u,indices]))
	#os._exit(0)

	for i in range(H):
		for j in range(W):
			#if(image[i][j]>100 and image[i][j]<140):
			#	img[i][j] = 128
			if(image[i][j]>Threshold):
				img[i][j]=255
			else:
				img[i][j]=0

	#cv2.imshow("Segment_using_histogram.jpg",img)
	cv2.imwrite('segment.jpg',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()


kernel = [-1,-1,-1],[-1,8,-1],[-1,-1,-1]
kernel = np.array(kernel)
#print(kernel)


cwd = os.getcwd()
path = os.path.join(cwd,'original_imgs')
image = cv2.imread(os.path.join(path,'point.jpg'),0)
image = cv2.Laplacian(image,cv2.CV_32F)


output = point_detection(image)
output = (output* 255).astype("uint8")
cv2.imwrite('porosity.png',output)

cwd = os.getcwd()
path = os.path.join(cwd,'original_imgs')
seg_image=cv2.imread(os.path.join(path,'segment.jpg'),0)
segmentation(seg_image)

