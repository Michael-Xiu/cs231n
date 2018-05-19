# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 22:10:19 2018

@author: Michael
"""

import CIFAR
import L3_SVM
import time

time_start=time.time()


#train_SVM(100,900,200) # (num_of_training_Pictures,times,Thresholdvalue):

#[W,F]=L3_SVM.train_SVM(100,1000,200)       #(num_of_training_Pictures,Loss_MAX,Thresholdvalue):
[W,F]=L3_SVM.train_SVM_improved(1500,8000,200)
[a,b,c]=L3_SVM.test_SVM(W,10000)


time_end=time.time()
print('totally cost',time_end-time_start)