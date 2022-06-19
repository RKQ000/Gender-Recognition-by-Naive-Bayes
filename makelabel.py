import numpy as np
import os
import random

# 读取有效数据的名字
files = os.listdir("./png")
new_list = []
for fileName in files:
    new_list.append(fileName[0:4])

# 读取原来的数据和标签
train_list = np.load('./train_data.npy')
valid_list = np.load('./test_data.npy')
train_list = train_list.tolist()
valid_list = valid_list.tolist()
total_list = train_list + valid_list

train_labels = np.load('./lable_age_DR.npy')
valid_labels = np.load('./lable_age_DS.npy')
train_labels = train_labels.tolist()
valid_labels = valid_labels.tolist()
total_labels = train_labels + valid_labels

new_label = []
print(len(new_list))
print(len(total_list))
for data in new_list:
    for i in range(len(total_list)):
        if int(data) == total_list[i]:
            new_label.append(total_labels[i])

print(len(new_label))

data_and_label = []
for i in range(len(new_label)):
    data_and_label.append([new_list[i],new_label[i]])
    
random.shuffle(data_and_label)

#total_list[0][1]
total_data_list = []
total_labels_list = []
for i in range(len(data_and_label)):
    total_data_list.append(data_and_label[i][0]) 
    total_labels_list.append(data_and_label[i][1])
    
np.save('shuffled_label.npy',total_labels_list)
np.save('shuffled_data.npy',total_data_list)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
for i in range(len(total_labels_list)):
    if total_labels_list[i]==0:
        class0 = class0 + 1
    elif total_labels_list[i]==1:
        class1 = class1 + 1
    elif total_labels_list[i]==2:
        class2 = class2 + 1
    elif total_labels_list[i]==3:
        class3 = class3 + 1