
import os
import random

data = []
split_size = 1
dirPath = '../datasets/HWDB1.1/train_data/'
folders = os.listdir(dirPath)
for folder in folders:
	# if int(folder) not in [i for i in range(0, 100)]:
	# 	continue
	curr_path = dirPath+folder
	image_list = os.listdir(curr_path)
	for image in image_list:
		data.append("%s %s"%(curr_path+'/'+image, str(int(folder))))

random.shuffle(data)

with open("data/train_data.txt", 'w') as f:
	for s in data:
		f.write("%s\n"%s)

# with open("data/train_data.txt", 'w') as f:
# 	for s in data[:int(split_size*len(data))]:
# 		f.write("%s\n"%s)
# with open("data/test_data.txt", 'w') as f:
# 	for s in data[int(split_size*len(data)):]:
# 		f.write("%s\n"%s)