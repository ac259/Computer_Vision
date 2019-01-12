import numpy as np
import math
import cv2
import os
import glob

user = os.path.expanduser('~')
path = 'Desktop/Computer_Vision_and_Image_Processing'
path = os.path.join(user,path)
def generate_kernel(sigma):
        coeff = 1/(2*math.pi*sigma*sigma)
        b = []
        for j in range(3,-4,-1):
            for i in range(-3,4):
                a = coeff*(math.exp(-((i*i +j*j)/(2*sigma*sigma))))
                b.append(a)
        return b

def create_zero_array(zero_array,image_height,image_width):
    zero_array = [[0 for x in range(image_height)] for y in range(image_width )]
    zero_array = np.array(zero_array)
    return zero_array


def Convolve(kernel,output,orig_image,image_with_padding):
    for x in range(orig_image.shape[1]):
        for y in range(orig_image.shape[0]):
            output[y,x] = (kernel*image_with_padding[y:y+7,x:x+7]).sum()
    return output
gauss = []
def octave(octave, image_number):
    global list_image_number
    for i in octave:
        gauss_kernel = generate_kernel(i)
        gauss_kernel = np.array(gauss_kernel)
        gauss_kernel = np.reshape(gauss_kernel,(7,7))
        sum = np.sum(gauss_kernel)
        gauss_kernel = np.divide(gauss_kernel,sum)

        img_path = 'task2_'+str(image_number)+".jpg"
        print(os.path.join(path,img_path))
        img_path = os.path.join(path,img_path)
        image = cv2.imread(img_path,0)
        orig_image = image
        image_height = image.shape[1]
        image_width = image.shape[0]

        # Padding the image
        image_with_padding = [[0 for x in range(image_height +6)] for y in range(image_width +6)]
        image_with_padding = np.array(image_with_padding)
        # We need to offset the value - x and y are 2
        x=3
        y=3

        image_with_padding[x:image.shape[0]+x, y:image.shape[1]+y] = image
        gauss_output = []
        gauss_output = create_zero_array(gauss_output,image_height,image_width)
        gauss_output = Convolve(gauss_kernel,gauss_output,orig_image,image_with_padding)
        gauss_output = gauss_output.astype("double")
        gauss.append(gauss_output)

        #cv2.imshow("Gauss-output",gauss_output.astype('uint8'))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imwrite("Gauss-output-"+str(list_image_number.index(image_number)+1)+"-"+str(octave.index(i))+".jpg",gauss_output)
        #print(image_with_padding.shape)
        #print(image_with_padding)


    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

first_octave = [(1/math.sqrt(2)),1,math.sqrt(2),2,(2*math.sqrt(2))]
second_octave = [math.sqrt(2),2,(2*math.sqrt(2)),4,(4*math.sqrt(2))]
third_octave = [(2*math.sqrt(2)),4,(4*math.sqrt(2)),8,(8*math.sqrt(2))]
fourth_octave = [(4*math.sqrt(2)),8,(8*math.sqrt(2)),16,(16*math.sqrt(2))]
list_octave = first_octave,second_octave, third_octave, fourth_octave
list_image_number = 1,2,4,8

for octaves,image_number in zip(list_octave,list_image_number):
    octave(octaves,image_number)
# U have now all the octave values in the list - now slice it and put it into the respective lists!
gauss_first_octave = []
gauss_second_octave = []
gauss_third_octave = []
gauss_fourth_octave = []
gauss_list = gauss_first_octave,gauss_second_octave,gauss_third_octave,gauss_fourth_octave

y = 0
i = 0
while (y<20 and i<4):
    temp = []
    temp= gauss[y:y+5]
    for item in temp:
        gauss_list[i].append(item)
    
    y = y+5
    i = i+1

dog_output = []
dog_first_octave = []
dog_second_octave = []
dog_third_octave = []
dog_fourth_octave = []
dog_list = dog_first_octave,dog_second_octave,dog_third_octave,dog_fourth_octave

def dog_gen(list_gauss):
    for i in range(0,4):
        dog_img = list_gauss[i]-list_gauss[i+1]
        dog_output.append(dog_img)

for gauss in gauss_list:        
    dog_gen(gauss)

y = 0
i = 0
while (y<16 and i<4):
    temp = []
    temp= dog_output[y:y+4]
    for item in temp:
        dog_list[i].append(item)
    y = y+4
    i = i+1
'''
for item in dog_list:
    for dog in item:
        cv2.imshow('DoG's,dog)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
'''
#print(dog_first_octave[2])


position_of_max_term = []
all_roi1 = []
key_points = []
flag = True

for x in range(0,dog_first_octave[k].shape[1]-1,3):
        for y in range(0,dog_first_octave[k].shape[0]-1,3):
            
            #print(k)
            dog10=dog_first_octave[k] 
            dog11 =dog_first_octave[k+1] 
            dog12 =dog_first_octave[k+2] 
            roi1 = dog11[y:y+3,x:x+3]
            max_min_point = roi1[1,1]
            
            all_roi1.append(roi1[1,1])
            orig =cv2.imread(r'C:\Users\ramji\Desktop\Computer_Vision_and_Image_Processing\task2.jpg',0)
            copy = orig
            max_min_point = np.max(roi1)
            roi0 = dog10[y:y+3,x:x+3]
            roi2 = dog12[y:y+3,x:x+3]
           
            if(roi1.shape[0] == 3 and roi1.shape[1] == 3):
                for i in range(roi0.shape[1]):
                    for j in range(roi0.shape[0]):
                        if(roi0[i,j]>max_min_point or roi2[i,j]>max_min_point or roi1[i,j]>max_min_point):
                            #print("Discard this point")
                            flag = False      
                        else:           
                            key_points.append(max_min_point)
                            if(max_min_point == dog11[y+1,x+1]):
                                
                                position_of_max_term.append([y+1,x+1])
                                

orig = cv2.imread(r'C:\Users\ramji\Desktop\Computer_Vision_and_Image_Processing\task2.jpg',0)
#print(position_of_max_term)
y,x = zip(*position_of_max_term)
#y,x = position_of_max_term
print(x)
print(y) 
os._exit(0)
def plot_point(img,key_point_list):
    for i in range(len(key_point_list)):
        y = key_point_list[i][0]
        print(y)
        x = key_point_list[i][1]
        print(x)
        img[y][x] = 255
        return img
key_point = plot_point(orig,position_of_max_term)
cv2.imshow('keypoint',key_point)
cv2.waitKey(0)
cv2.destroyAllWindows()

