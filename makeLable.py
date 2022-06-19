"""
Thus to the data has some empty image, I delete them and remake the lable list
"""
import os
import numpy as np

files = os.listdir("./png")
new_list = []
for fileName in files:
    new_list.append(fileName[0:4])

train_list = np.load('./train_data_list.npy')
valid_list = np.load('./test_data_list.npy')
train_list = train_list.tolist()
valid_list = valid_list.tolist()
total_list = train_list + valid_list

train_labels = np.load('./train_labels_list.npy')
valid_labels = np.load('./test_labels_list.npy')
train_labels = train_labels.tolist()
valid_labels = valid_labels.tolist()
total_labels = train_labels + valid_labels

new_lables = []
for data in new_list:
    new_lables.append(total_labels[total_list.index(data)])

np.save('new_lables.npy',new_lables)
np.save('new_data.npy',new_list)
