import os
import glob
import shutil


# Shutil is used to move the files.

''' 
This is a template - you can modify the code to automate according to your needs
This program is used to run the programs ( all the programs you wish for)
and automate the process. This will create a folder and transfer all the output to 
that folder..
'''
type = ('*.jpg','*.png') # You can add any other file formats
files_grabbed = []


# This is used to run the first python script..
os.system('python name_of_the_program1.py')
os.mkdir('name_of_dir1')

source = os.getcwd()
dest = os.path.join(source,'name_of_the_program1')
for files in glob.glob(os.path.join(os.getcwd(),'*.jpg')):
	print(files)
	shutil.move(files, dest)

os.system('python name_of_the_program2.py')
os.mkdir('name_of_dir2')
dest = os.path.join(source,'name_of_the_program2')
for files in glob.glob(os.path.join(os.getcwd(),'*.jpg')):
	print(files)
	shutil.move(files, dest)

for files in glob.glob(os.path.join(os.getcwd(),'*.png')):
	print(files)
	shutil.move(files, dest)

os.system('python name_of_the_program3.py')
os.mkdir('name_of_dir3')
dest = os.path.join(source,'name_of_the_program3')
for files in glob.glob(os.path.join(os.getcwd(),'*.jpg')):
	print(files)
	shutil.move(files, dest)