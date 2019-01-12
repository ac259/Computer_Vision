import cv2
import numpy as np
import os

def dilation_erosion(img,flag):
    h, w = np.shape(img)
    new = img.copy()
    #print(new.shape)
    #print(flag)
    for pixel_y in range(0, h):
        for pixel_x in range(0, w):
            # extract 5x5 pixel area
            roi = img[max(0, pixel_y - 2): min(h, pixel_y + 3),
                      max(0, pixel_x - 2): min(w, pixel_x + 3)]
                      
            if flag == 'd':
                new[pixel_y][pixel_x] = np.amax(roi)
                #print("Inside dilation condition")
            elif flag == 'e':
                new[pixel_y][pixel_x] = np.amin(roi)
                #print("Inside erosion condition")
            else:
                break;
    return new


cwd = os.getcwd()
path = os.path.join(cwd,'original_imgs')
image = cv2.imread(os.path.join(path,"noise.jpg"),0)


# Opening + Closing
img2 = dilation_erosion(image.copy(),'e')
img2 = dilation_erosion(img2,'d')
img2 = dilation_erosion(img2,'d')
open_close = dilation_erosion(img2,'e')
#cv2.imshow('Opening followed by Closing',open_close)
cv2.imwrite('res_noise1.jpg',open_close)

# Closing + Opening
img2 = dilation_erosion(image.copy(),'d')
img2 = dilation_erosion(img2,'e')
img2 = dilation_erosion(img2,'e')
close_open = dilation_erosion(img2,'d')
#cv2.imshow('Closing followed by Opening',close_open)
cv2.imwrite('res_noise2.jpg',close_open)

# Closing + Opening + Closing
img2 = dilation_erosion(image.copy(),'d')
img2 = dilation_erosion(img2,'e')
img2 = dilation_erosion(img2,'e')
img2 = dilation_erosion(img2,'d')
img2 = dilation_erosion(img2,'d')
close_open_close = dilation_erosion(img2,'e')
#cv2.imshow('Closing followed by Opening followed by Closing',close_open_close)
cv2.imwrite('res_noise2.jpg',close_open)

#boundary 1
img2 = dilation_erosion(open_close,'e')
img_e = dilation_erosion(img2,'e')
boundary_1 = img2 - img_e
#cv2.imshow('Boundary_1',boundary_1)
cv2.imwrite('res_bound1.jpg',boundary_1)

#boundary 2
img2 = dilation_erosion(close_open,'d')
img_e = dilation_erosion(img2,'e')
boundary_2 = img2 - img_e
#cv2.imshow('Boundary 2',boundary_2)
cv2.imwrite('res_bound2.jpg',boundary_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
