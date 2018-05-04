#【1】导入常用的库
import os
import cv2
import random
import pandas as pd
import numpy as np
import tensorflow as tf

import  matplotlib.pyplot as plt
from matplotlib import ticker #???
import seaborn as sns

from keras.models import Sequential #???
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D,Dense, Activation
from keras.optimizers import RMSprop #???
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils #???

#【2】准备数据
TRAIN_DIR = 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_small'
TEST_DIR = 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\test_samll'

#读取猫，狗的数据
#参考：http://www.runoob.com/python/os-listdir.html -- Python os.listdir() 方法，只支持在 Unix, Windows 下使用。
train_dogs = [(TRAIN_DIR, 1) for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [(TRAIN_DIR, 0) for i in os.listdir(TRAIN_DIR) if 'cat' in i]

#放入训练集
test_images = [(TEST_DIR, -1) for i in os.listdir(TEST_DIR)]

#把train 猫+狗 合成一个数据集
train_images = train_dogs[:20] + train_cats[:20]
random.shuffle(train_images)
test_imagess = test_images[:10]

print('*********************')

#使用OpenCV读取图片,统一标准，把所有的图片resize进60*64方格
ROWS = 64
COLS = 64
def read_image(tuple_set):
    file_path = tuple_set[0]
    label = tuple_set[1]
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    print(file_path)
    print('i am here')
    return cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC), label

#预处理图片，把图片变为numpy数组
CHANNELS = 3 #代表RGB三个颜色通道
def prep_data(images):
    no_images = len(images)
    data = np.ndarray((no_images, CHANNELS, ROWS, COLS), dtype=np.uint8)
    labels = []

    for i, image_file in enumerate(images):
        image, label = read_image(image_file)
        data[i] = image_file.T
        labels.append(label)
    return data, labels
print('*******2**************')

x_train, y_train = prep_data(train_images)
x_test, y_shit = prep_data(test_images)

print(x_train.shape)
print(x_test.shape)

