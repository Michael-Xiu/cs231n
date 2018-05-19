# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:34:53 2018

@author: Michael
"""


import numpy as np
import CIFAR
#import time


#进行训练集数据的导入，以及W矩阵的初始化
def select_training_set(num_of_Pictures):
    W1=np.random.random([10,3072])/1000 #初始化W矩阵
    W2=np.random.random([10,10])/1000
    b1=np.zeros([10,1])
    b2=np.zeros([10,1])
    train_set=CIFAR.train_n(num_of_Pictures)
    X=train_set['data'].T  #初始化X矩阵
    X=X/256
    return W1,W2,b1,b2,X,train_set

def Neural_Network(W1,W2,X,b1,b2,reg,train_set,num_of_Pictures):  #max作为Loss Function
    n=num_of_Pictures
    #第一层：ReLu
    F1=W1.dot(X)+b1  #引入偏置
    h_array=np.maximum(0,F1)
    
    #第二层：Softmax
    F2=W2.dot(h_array)+b2 
    Loss_array=np.zeros([1,n])
    softmax_output=np.zeros([10,n])
    for i in range(n):
        Correct_label=train_set['labels'][i]
        softmax_output[:,i]=np.exp(F2[:,i])/np.sum(np.exp(F2[:,i]))
        Loss_array[0][i]=softmax_output[Correct_label][i] #运用softmax函数 -log10(exp(correct)/sum(exp(each)))
    Loss_total=-np.log(np.sum(Loss_array))
    Loss_total/=n
    Loss_total+=0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2)) #正则项

    Gradient_F2=softmax_output.copy()  #Softmax梯度，若不是label位置，则等于本身
    Gradient_F2[train_set['labels'][0:n],range(n)]-=1  #若是label位置，则等于本身-1
    Gradient_F2/=n
    
    Gradient_W2=Gradient_F2.dot(h_array.T)+reg*W2
    Gradient_b2=np.sum(Gradient_F2,axis=1).reshape(10,1)
    
    Gradient_h=W2.T.dot(Gradient_F2)  
    Gradient_ReLu=(h_array>0)*Gradient_h
    Gradient_W1=Gradient_ReLu.dot(X.T)+reg*W1
    Gradient_b1=np.sum(Gradient_ReLu,axis=1).reshape(10,1)
    
    return {'Loss_total':Loss_total,
            'Gradient_W1':Gradient_W1,
            'Gradient_b1':Gradient_b1,
            'Gradient_W2':Gradient_W2,
            'Gradient_b2':Gradient_b2,
            'Gradient_h':Gradient_h,
            'Gradient_ReLu':Gradient_ReLu,
            'softmax_output':softmax_output,
            'Loss_array':Loss_array}
    
def train_Neural_Network(num_of_Pictures,reg=1e-5,Step_size=1e-3,num_iters=100):  #SGD随机梯度下降
    Loss_history=[]  #平均损失函数
    train_acc_history=[]  #平均训练准确率
    val_acc_history=[]  #平均预测准确率
    [W1,W2,b1,b2,X,train_set]=select_training_set(num_of_Pictures)
    for j in range(5):
        for i in range(num_iters): #每群样本迭代多少次
            Result=Neural_Network(W1,W2,X,b1,b2,reg,train_set,num_of_Pictures)  #max作为Loss Function
            Loss_history.append(Result['Loss_total'])
            #更新参数
            W1+=-Step_size*Result['Gradient_W1']
            W2+=-Step_size*Result['Gradient_W2']
            b1+=-Step_size*Result['Gradient_b1']
            b2+=-Step_size*Result['Gradient_b2']
        [percent,true_num,num,F2,Labels_Predict]=test_Neural_Network(W1,W2,b1,b2,10000,1) 
        val_acc_history.append(percent)
        print('Loss is %f'%(percent))
    return val_acc_history,Loss_history,{'W1':W1,'W2':W2,'b1':b1,'b2':b2},F2,Labels_Predict,Result
    
def test_Neural_Network(W1,W2,b1,b2,num,Test_batch=1):
    test_set=CIFAR.test_minibatch(num)
    X=test_set['data'].T
    X=X/256
    h_array=np.maximum(0,W1.dot(X)+b1)
    F2=W2.dot(h_array)+b2
    Labels_Predict=np.argmax(F2,axis=0)    #取预测矩阵F的每列最大值，作为该图像的预测label
    Labels_True_list=test_set['labels']
    Labels_True=np.array(Labels_True_list,'int')#将list转成array进行比较，用==会简化
    comp=(Labels_Predict==Labels_True)          #生成bool矩阵，判断对应label是否相等
    true_num=np.sum(comp)
    percent=true_num/num
    return percent,true_num,num,F2,Labels_Predict
    
    
    