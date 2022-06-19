"""
shuffle the dataset, select the train and test data randomly
"""


import numpy as np
import random

total_data_list = np.load('./new_data.npy')
total_data_list = total_data_list.reshape(3956,1)
total_labels_list = np.load('./new_labels.npy')
total_labels_list = total_labels_list.reshape(3956,1)

total_list = []
for i in range(len(total_data_list)):
    total_list.append([total_data_list[i][0],total_labels_list[i][0]])

# 随机打乱数据
random.shuffle(total_list)

#total_list[0][1]
for i in range(len(total_list)):
    total_data_list[i] = total_list[i][0]
    total_labels_list[i] = total_list[i][1]
np.save('shuffled_labels.npy',total_labels_list)
np.save('shuffled_data.npy',total_data_list)
    
test_labels_list = total_labels_list[3000:3956]
    
femaleNum = 0
for i in range(len(test_labels_list)):
    femaleNum = femaleNum + total_labels_list[i][0]
print(femaleNum)