UBIT = 'ac259';import numpy as np;np.random.seed(sum([ord(c) for c in UBIT]))
import matplotlib.pyplot as plt 
import cv2
import os

def plotter(data,facecolor='none',marker='^'):
	x_val,y_val = zip(*data)
	plt.scatter(x_val,y_val,marker=marker,c=facecolor, edgecolors='none')
	for x,y in zip(x_val, y_val): 
		z = str(x)+','+str(y)                                      
		plt.text(x,y,'(%s)'%str(z),fontsize=6) 

data = [(5.9,3.2),(4.6,2.9),(6.2,2.8),(4.7,3.2),(5.5,4.2),(5.0,3.0),(4.9,3.1),(6.7,3.1),(5.1,3.8
),(6.0,3.0)]

x = [5.9,4.6,6.2,4.7,5.5,5.0,4.9,6.7,5.1,6.0]
y = [3.2,2.9,2.8,3.2,4.2,3.0,3.1,3.1,3.8,3.0]

centroids =[(6.2,3.2),(6.6,3.7),(6.5,3.0)]
c_x,c_y = zip(*centroids)

C = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
plt.scatter(c_x, c_y,c= C/255.0)
for x,y in zip(c_x,c_y): 
		z = str(x)+','+str(y)                                      
		plt.text(x,y,'(%s)'%str(z),fontsize=6)

def list_to_npArray(vector1, vector2):
    '''convert the list to numpy array'''
    if type(vector1) == list:
        vector1 = np.array(vector1)
    if type(vector2) == list:
        vector2 = np.array(vector2)
    return vector1, vector2

def euclidean(vector1, vector2):
    ''' using numpy.linalg.norm to calculate the euclidean distance. '''
    vector1, vector2 = list_to_npArray(vector1, vector2)
    distance = np.linalg.norm(vector1-vector2, 2, 0)
    # the third argument "0" means the column, and "1" means the line.
    return distance

def distance_2_list(vect,cent_list):
	for data_point in data:
		dist = euclidean(data_point,vect)
		cent_list.append(dist)

def calulate_cluster(vect1,vect2,vect3):
	vector = vector1,vector2,vector3

	centroid1 = centroid2 = centroid3 = []
	centroid_list = [centroid1,centroid2,centroid3]
		
	c_list = []
	for vect in vector:
			distance_2_list(vect,c_list)

	centroid1 = c_list[0:10]
	
	centroid2 = c_list[10:20]
	
	centroid3 = c_list[20:30]
	
	cluster1= []
	cluster2 = []
	cluster3 = []

	for i,j,k in zip(centroid1,centroid2,centroid3):
		
		if(i<j and i<k):
			
			cluster1.append(centroid1.index(i))
			
		elif(j<i and j<k):
			
			cluster2.append(centroid2.index(j))
			
		else:
			
			cluster3.append(centroid3.index(k))
		
	cluster_with_points1 = []
	cluster_with_points2 = []
	cluster_with_points3 = []

	for index in cluster1:
		cluster_with_points1.append(data[index])

	plotter(cluster_with_points1,'red')

	for index in cluster2:
		cluster_with_points2.append(data[index])
	plotter(cluster_with_points2,'green')
	
	for index in cluster3:
		cluster_with_points3.append(data[index])
	plotter(cluster_with_points3,'blue')

	return cluster_with_points1,cluster_with_points2,cluster_with_points3


def plot_updated_centers(cp1,cp2,cp3):

	new_cluster_centroid_1 = np.mean(np.asarray(cp1,dtype ='float64'),axis = 0)
	c_x,c_y = new_cluster_centroid_1
	
	C = np.array([255, 0, 0])
	plt.scatter(c_x, c_y,c= 'red', edgecolors='none') 
	z_new = str("{0:.2f}".format(c_x))+','+str("{0:.2f}".format(c_y)) 
	                                   
	plt.text(c_x,c_y,'(%s)'%str(z_new),fontsize=6)

	new_cluster_centroid_2 = np.mean(np.asarray(cp2,dtype ='float64'),axis = 0)
	c_x,c_y = new_cluster_centroid_2
	
	C = np.array([0, 255, 0])
	plt.scatter(c_x, c_y,c= 'green', edgecolors='none') 
	z_new = str("{0:.2f}".format(c_x))+','+str("{0:.2f}".format(c_y)) 
	                                   
	plt.text(c_x,c_y,'(%s)'%str(z_new),fontsize=6)

	new_cluster_centroid_3 = np.mean(np.asarray(cp3,dtype ='float64'),axis = 0)
	c_x,c_y = new_cluster_centroid_3
	
	C = np.array([0, 0, 255])
	plt.scatter(c_x, c_y,c= 'blue', edgecolors='none') 
	z_new = str("{0:.2f}".format(c_x))+','+str("{0:.2f}".format(c_y))  
	                                    
	plt.text(c_x,c_y,'(%s)'%str(z_new),fontsize=6)
	return new_cluster_centroid_1,new_cluster_centroid_2,new_cluster_centroid_3
	

vector2 = np.array((6.6,3.7))
vector1 = np.asarray((6.2,3.2))
vector3 = np.asarray((6.5,3.0))
cp1,cp2,cp3 = calulate_cluster(vector1,vector2,vector3)
print("----------------------------------------------------------------")
print("Cluster 1: ",cp1,"\nCluster 2: ",cp2,"\nCluster 3:",cp3)
print("----------------------------------------------------------------")

plt.savefig('task3_iter1_a',dp=300,papertype='a4')


plt.gcf().clear()
cp1,cp2,cp3 = calulate_cluster(vector1,vector2,vector3)

new_cluster_centroid_1,new_cluster_centroid_2,new_cluster_centroid_3 =plot_updated_centers(cp1,cp2,cp3)
plt.savefig('task3_iter1_b',dp=300,papertype='a4')

plt.gcf().clear()
vector1 = np.asarray(new_cluster_centroid_1)
vector2 = np.asarray(new_cluster_centroid_2)
vector3 = np.asarray(new_cluster_centroid_3)
cp1_2,cp2_2,cp3_2 = calulate_cluster(vector1,vector2,vector3)
print("Cluster 1: ",cp1_2,"\nCluster 2: ",cp2_2,"\nCluster 3:",cp3_2)
print("----------------------------------------------------------------")

new_cluster_centroid_1,new_cluster_centroid_2,new_cluster_centroid_3 =plot_updated_centers(cp1,cp2,cp3)
plt.savefig('task3_iter2_a',dp=300,papertype='a4')
plt.gcf().clear()
cp1_2,cp2_2,cp3_2 = calulate_cluster(vector1,vector2,vector3)
print("Cluster 1: ",cp1_2,"\nCluster 2: ",cp2_2,"\nCluster 3:",cp3_2)
print("----------------------------------------------------------------")
new_cluster_centroid_1,new_cluster_centroid_2,new_cluster_centroid_3 =plot_updated_centers(cp1_2,cp2_2,cp3_2)
plt.savefig('task3_iter2_b',dp=300,papertype='a4')

