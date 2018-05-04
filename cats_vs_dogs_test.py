import random
import pandas as pd
import numpy as np
import tensorflow as tf
import os

import  matplotlib.pyplot as plt
from matplotlib import ticker #???
import seaborn as sns

from keras.models import Sequential #???
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D,Dense, Activation
from keras.optimizers import RMSprop #???
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils #???

#【1】获取图片文件名 + 对应的标签
train_dir = 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_small'
def get_files(file_dir):
    '''
    :param file_dir: file directory
    :return: list of images and labels
    '''
    images = []
    labels = []
    for file in os.listdir(file_dir): #图片名字file = cat.3700.jpg
        name = file.split(sep='.') #分割开，组成list，name = ['cat', '3700', 'jpg']
        if name[0] == 'cat':   #判断第一个元素是否是cat
            images.append(file_dir + file) #把文件路径 + 图片路径
            labels.append(0) #把属于cat的图片归类到0，这一栏中
        else:
            images.append(file_dir + file) #把dog归类到这里
            labels.append(1)
    return images, labels

#输出的结果：
# ['C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3700.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3701.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3702.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3703.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3704.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3705.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3706.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3707.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3708.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3709.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3710.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3711.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3712.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3713.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3714.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3715.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3716.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3717.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3718.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3719.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3720.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3721.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3722.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3723.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3724.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3725.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3726.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3727.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3728.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3729.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3730.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3731.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3732.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3733.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3734.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3735.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3736.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3737.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3738.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3739.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3740.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3741.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3742.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3743.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3744.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3745.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3746.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3747.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3748.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3749.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smallcat.3750.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4000.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4001.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4002.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4003.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4004.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4005.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4006.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4007.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4008.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4009.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4010.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4011.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4012.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4013.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4014.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4015.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4016.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4017.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4018.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4019.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4020.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4021.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4022.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4023.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4024.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4025.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4026.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4027.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4028.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4029.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4030.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4031.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4032.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4033.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4034.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4035.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4036.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4037.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4038.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4039.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4040.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4041.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4042.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4043.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4044.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4045.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4046.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4047.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4048.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4049.jpg', 'C:\\Users\\xiaochen.liu\\Desktop\\test_1\\cat_dog\\train_smalldog.4050.jpg']
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


#【2】分批次读取图片，图片太多，读取进内存，会造成内存不够用（我这里偷懒一共用了50张cat，50张dog，50张）
#利用tensorflow中的tf.train.slice_input_producer函数，利用队列思想
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    :param image: list type
    :param label:  list type
    :param image_W: image width
    :param image_H: image height
    :param batch_size: batch size
    :param capacity: the maximum elements in quene
    :return:
    image_batch: 4D tensor[batch_size, width, height, 3], dtype = tf.float32
    label_batch: 1D tensor [batch_size], dtype = tf.in32
    '''
    #将python的list数据类型转换为tensorflow的数据类型
    image = tf.convert_to_tensor(image, dtype=tf.string)

