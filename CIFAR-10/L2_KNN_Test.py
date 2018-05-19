# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 20:46:15 2018

@author: Michael
"""

import numpy as np
import L2_KNN

#data=L2_KNN.load_file('data_batch_1')

train=L2_KNN.train()  #进行训练
test=L2_KNN.load_file('test_batch')  #读取test文件
test_label=test['labels'][0:1000]
pre_result=L2_KNN.predict_all(train,test)  #对test进行预测
precent=L2_KNN.comp_result(test_label,pre_result)   #计算预测的结果