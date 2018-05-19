# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 22:02:53 2018

@author: Michael

used to load the training set and testing set. we can choose how much we load
"""

import numpy as np
import pickle

#读取CIFAR—10
def load_file(filename):  #读取CIFAR-10里其中一个文件的数据
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

#生成训练集
def train_n(n):   #加载n个图像
    dict1=load_file('data_batch_1')
    dict2=load_file('data_batch_2')
    dict3=load_file('data_batch_3')
    dict4=load_file('data_batch_4')
    dict5=load_file('data_batch_5')
    dict_sum=dict1
    #创建一个包括所有数据集的dict，data为二维数组，进行相应扩展
    dict_sum['data']=np.concatenate((dict1['data'],dict2['data'],dict3['data'],dict4['data'],dict5['data']))
    dict_sum['data']=np.array(dict_sum['data'],float)
    #labels是一个list，同样进行扩展
    dict_sum['labels'].extend(dict2['labels'])
    dict_sum['labels'].extend(dict3['labels'])
    dict_sum['labels'].extend(dict4['labels'])
    dict_sum['labels'].extend(dict5['labels'])
    dict_sum['filenames'].extend(dict2['filenames'])
    dict_sum['filenames'].extend(dict3['filenames'])
    dict_sum['filenames'].extend(dict4['filenames'])
    dict_sum['filenames'].extend(dict5['filenames'])
    dict_mini=dict_sum
    dict_mini['data']=dict_sum['data'][0:n]
    dict_mini['filenames']=dict_sum['filenames'][0:n]
    dict_mini['labels']=dict_sum['labels'][0:n]
    return dict_mini

def test_minibatch(n):  #加载n个测试图像 
    dict1=load_file('test_batch')
    dict1['data']=np.array(dict1['data'],float)
    dict_mini=dict1
    dict_mini['data']=dict1['data'][0:n]
    dict_mini['filenames']=dict1['filenames'][0:n]
    dict_mini['labels']=dict1['labels'][0:n]
    return dict_mini



