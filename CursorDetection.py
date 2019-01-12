import cv2
import numpy as np
import glob
import imutils
import os

user = os.path.expanduser('~')
path = os.getcwd()
path = os.path.join(user,path)
imagePath = r'task3\*.jpg'

template_path = r'task3\templates\*.jpg'
print(os.path.join(path,template_path))
templates = []
for temp in glob.glob(os.path.join(path,template_path)):
	template = cv2.imread(temp)
	#cv2.imshow('orig-temp',template)
	laplacian_template = cv2.Laplacian(template,cv2.CV_32F)
	templates.append(laplacian_template)
	#cv2.imshow('laplacian_template',laplacian_template)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()


for imagePath in glob.glob(os.path.join(path,imagePath)):
	image = cv2.imread(os.path.join(path,imagePath))
	
	blur = cv2.GaussianBlur(image,(3,3),0)
	laplacian_output = cv2.Laplacian(blur,cv2.CV_32F)
	for temp in templates:
		gray_temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
		w, h = gray_temp.shape[::-1]

		res = cv2.matchTemplate(laplacian_output,temp,cv2.TM_CCOEFF_NORMED)
		threshold = 0.45
		loc = np.where( res >= threshold)

		for pt in zip(*loc[::-1]):
		    cv2.rectangle(image,pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

	#cv2.imwrite('laplacian.jpg',laplacian_output)
	#cv2.imshow('laplacian template',laplacian_template)
	i = 0
	i = i+1
	#cv2.imwrite('laplacian-template.jpg',laplacian_template)
	cv2.imshow('Detected',image)
	#cv2.imwrite('Detected'+str(image)+'.jpg',image)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()