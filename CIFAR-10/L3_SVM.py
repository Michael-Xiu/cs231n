# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 22:16:44 2018

@author: Michael
"""

import numpy as np
import CIFAR
import time


#进行训练集数据的导入，以及W矩阵的初始化
def select_training_set(num_of_Pictures):
 W=np.zeros([10,3072]) #初始化W矩阵
 train_set=CIFAR.train_n(num_of_Pictures)
 X=train_set['data'].T  #初始化X矩阵
 return W,X,train_set

#进行前向计算
def Forward(W,X,train_set,num_of_Pictures,Thresholdvalue):  #max作为Loss Function
    F=W.dot(X)
    n=num_of_Pictures
    #分两部构建正确的参考矩阵Correct_array，同时计算误差矩阵Loss_array
    Correct_array=np.zeros([10,n])
    Loss_array=np.zeros([10,n])
    for i in range(n):  
        Correct_label=train_set['labels'][i]
        Correct_array[Correct_label][i]=1
        Loss_array[:,i]=np.maximum(0,F[:,i]-F[Correct_label][i]+Thresholdvalue)  #max(0,sj-sr+1)
        Loss_array[Correct_label][i]=0    #对于正确位置loss设为0
    Loss_total=np.sum(Loss_array)
    return Loss_array,Loss_total,num_of_Pictures,F  #F仅用来观看

#进行后向更新W
def Backward(W,X,Loss_array,train_set,num_of_Pictures):
    n=num_of_Pictures
    #Back propagation
    #Step1:求出F矩阵的梯度
    Gradient_F_above0_pos=np.array((Loss_array>0),float)  #找出Loss_array内元素进行max后取了sj-sr+1的位置
    Gradient_F=Gradient_F_above0_pos.copy()  #深复制避免相互影响
    for i in range(n):
        Correct_label=train_set['labels'][i] 
        Gradient_F[Correct_label][i]=-1*np.sum(Gradient_F_above0_pos[:,i])
    
    #Step2:求出W矩阵的梯度
    Gradient_W=np.zeros([10,3072])
    for i in range(10):
        for j in range(3072):
            Gradient_W[i][j]=np.sum(np.multiply(Gradient_F[i,:],X[j,:]))
    
    #Step3:修正W矩阵
    Step_size=0.0000001  #对于训练集为10时 0.00000001
    W=W-Step_size*Gradient_W
    return W

#用训练集，训练SVM模型
def train_SVM(num_of_training_Pictures,Loss_MAX,Thresholdvalue):  #Thresholdvalue为max函数的阈值
    n=num_of_training_Pictures
    [W,X,train_set]=select_training_set(n)
    Loss_total=1000000000
    count=0
    while(Loss_total>Loss_MAX):   #模型迭代到Loss_total小于预期值
        for i in range(50):
            [Loss_array,Loss_total,num_of_Pictures,F]=Forward(W,X,train_set,n,Thresholdvalue)  #传入train_set目的为取其label
            W=Backward(W,X,Loss_array,train_set,n)
            count=count+1
            print("Training times=%d, Loss=%f" % (count,Loss_total))
    return W,F
          

#用测试集，测试SVM模型       
def test_SVM(W,num_of_testing_Pictures):
    n=num_of_testing_Pictures
    test_set=CIFAR.test_minibatch(n)
    X_test=test_set['data'].T
    F_test=W.dot(X_test)    #用训练好的W和testing_batch相乘，判断预测效果的好坏
    Labels_Predict=np.argmax(F_test,axis=0)    #取预测矩阵F的每列最大值，作为该图像的预测label
    Labels_True_list=test_set['labels']
    Labels_True=np.array(Labels_True_list,'int')#将list转成array进行比较，用==会简化
    comp=(Labels_Predict==Labels_True)          #生成bool矩阵，判断对应label是否相等
    true_num=np.sum(comp)
    percent=true_num/n
    return percent,true_num,n
        
#SVM总的外部接口
def SVM(num_of_training_Pictures,num_of_testing_Pictures,training_times,Thresholdvalue):
    W_trained=train_SVM(num_of_training_Pictures,training_times,Thresholdvalue)
    Percentage_of_Correct_Classification=test_SVM(W_trained,num_of_testing_Pictures)
    return Percentage_of_Correct_Classification

def train_SVM_improved(num_of_training_Pictures,Loss_MAX,Thresholdvalue):
    #思想是例如需要训练10000个图像，则每次用50个进行一次不那么精确的迭代
    n=num_of_training_Pictures
    [W,X_all,train_set_all]=select_training_set(n)
    count=0
    times=int(n/50)  
    for i in range(times):
        time_start=time.time()
        Loss_total=100000000
        print("Section from %d to %d" %(50*i+1,50*(i+1)))
        X=X_all[:,50*i:50*(i+1)].copy()  #取出这一部分的X作为data
        train_set=train_set_all.copy()
        train_set['labels']=train_set_all['labels'][50*i:50*(i+1)].copy() #取出这一部分的label
        while(Loss_total>Loss_MAX):   #模型迭代到Loss_total小于预期值
            for j in range(10):
                [Loss_array,Loss_total,num_of_Pictures,F]=Forward(W,X,train_set,50,Thresholdvalue)
                W=Backward(W,X,Loss_array,train_set,50)
                count=count+1
                print("Training times=%d, Loss=%f" % (count,Loss_total))
        time_end=time.time()
        print('totally cost',time_end-time_start)
    return W,F
    
    
'''
  Gradient_F_above0_pos找出Loss_array内元素进行max后取了sj-sr+1的位置,进行从loss
往F反向传播时，这些位置对应的位置梯度会+1，同时正确位置的梯度会-1。正确位置的元素会
出现在其他9个元素内，其梯度最终为相加值。
'''

'''
备注：
  二位索引最好写在一个[]内
'''