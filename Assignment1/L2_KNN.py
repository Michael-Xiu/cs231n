# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 20:40:28 2018

@author: Michael

Homework: Lesson 2 of Stanford CNN
"""

import numpy as np
import pickle
import operator
from os import listdir

#读取CIFAR—10
def load_file(filename):  #读取CIFAR-10里其中一个文件的数据
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

#生成训练集
def train():
    dict1=load_file('data_batch_1')
    dict2=load_file('data_batch_2')
    dict3=load_file('data_batch_3')
    dict4=load_file('data_batch_4')
    dict5=load_file('data_batch_5')
    dict_sum=dict1
    #创建一个包括所有数据集的dict，data为二维数组，进行相应扩展
    dict_sum['data']=np.concatenate((dict1['data'],dict2['data'],dict3['data'],dict4['data'],dict5['data']))
    #labels是一个list，同样进行扩展
    dict_sum['labels'].extend(dict2['labels'])
    dict_sum['labels'].extend(dict3['labels'])
    dict_sum['labels'].extend(dict4['labels'])
    dict_sum['labels'].extend(dict5['labels'])
    return dict_sum

#对一幅图像进行预测
def predict_one(training_dict,testing_data): #traing_dict为一幅图像的数据，k为设置的最近邻个数
    training_dict_int=np.array(training_dict['data'],'int')  #将uint8转为int数组进行处理，防止溢出
    testing_data_int=np.array(testing_data,'int')
    a1=training_dict_int-testing_data_int
    a2=np.abs(a1)
    a3=np.sum(a2,axis=1)
    sort_index=np.argmin(a3)   #求出距离矩阵最小值对应的index，用index找对应label
    label_predict=training_dict['labels'][sort_index]
    return label_predict

#对所有图像进行预测
def predict_all(training_dict,testing_dict_all):
    result=[]
    num_of_training=np.size(testing_dict_all['filenames'])
    for i in range(1000):
        result_one=predict_one(training_dict,testing_dict_all['data'][i])
        result.append(result_one)
    return result

#比较预测结果
def comp_result(testing_dict,predict_result):  
    true_result=testing_dict#testing_dict['labels']
    true_result_array=np.array(true_result,'int')#将list转成array进行比较，用==会简化
    predict_result_array=np.array(predict_result,'int')
    comp=(true_result_array==predict_result_array)   #生成bool矩阵，判断对应label是否相等
    true_num=np.sum(comp)
    percent=true_num/np.size(predict_result)
    return percent
    
    
    
    
    