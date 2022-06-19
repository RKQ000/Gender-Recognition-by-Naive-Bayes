"""
this code extract the LBP feature of the image
"""

import skimage.feature
import numpy as np
import cv2
import PIL



total_list = np.load('./new_list.npy')


folderPath = './png/'
savePath = './LBP/'
radius = 1
n_point = radius * 8

for fileName in total_list:
    imagePath = folderPath + fileName
    image = cv2.imread(imagePath+'.png', 0)
    if image is None:
        continue
    LBPimage = skimage.feature.local_binary_pattern(image, n_point, radius, method='default')
    f_path = savePath + fileName + '.png'
    cv2.imwrite(f_path, LBPimage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# image = PIL.Image.fromarray(image, mode=None)
# LBPimage = PIL.Image.fromarray(LBPimage, mode=None)
# image.show()
# LBPimage.show()
