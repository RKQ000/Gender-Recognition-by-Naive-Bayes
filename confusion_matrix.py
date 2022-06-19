# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:34:58 2022

@author: RocketQI
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:45:39 2022

@author: RocketQI
"""
import torch 
import numpy as np
import os
import cv2
from util import *
from ResNet import *
from RegNet import *
from DenseNet import *
from MobileNetV2 import *
from EfficientNetV2 import *
from torchvision.transforms import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

cuda = 0
valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomCrop(64),
    transforms.ToTensor(),
])

data = np.load('./shuffled_data.npy')
labels = np.load('./shuffled_label.npy')

valid_list = data[3000:3956]
valid_labels = labels[3000:3956]

valid_loader = GetLoader(X=valid_list, y=valid_labels, batch_size=16, folder='./png/', transform=valid_transform, stage=0)
model = EfficientNet().cuda()
model.load_state_dict(torch.load('./save/EfficientNet.pt'))

with torch.no_grad():
    conf_matrix_test = torch.zeros(4, 4)
    for data, label in valid_loader:
        data = data.cuda().float()
        label = label.cuda()
        pred = model(data)
        conf_matrix_test = confusion_matrix(pred, label, conf_matrix_test)
        conf_matrix_test = conf_matrix_test.cpu()

conf_matrix_test = np.array(conf_matrix_test.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
print('混淆矩阵test:')
print(conf_matrix_test)

#CM = confusion_matrix(valid_labels, predic)
#plt.matshow(conf_matrix_test, cmap=plt.cm.Reds)
cm = conf_matrix_test
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
plt.colorbar()
labels_name = ['child','teen','adult','senior']
num_local = np.array(range(len(labels_name)))  
plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
plt.ylabel('True label')    
plt.xlabel('Predicted label')
for i in range(4):
   for j in range(4):
       data = cm[j][i]
       data = str(data)[0:5]
       plt.text(-0.4+i, 0.1+j, data, fontsize='x-large',color = 'red')

