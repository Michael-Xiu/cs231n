# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:04:43 2018

@author: Michael
"""

import numpy as np
import CIFAR
#import time


#进行训练集数据的导入，以及W矩阵的初始化
def select_training_set(num_of_Pictures):
 W1=np.random.random([10,3072])/1000 #初始化W矩阵
 W2=np.random.random([10,10])/1000
 b1=0
 b2=0
 train_set=CIFAR.train_n(num_of_Pictures)
 X=train_set['data'].T  #初始化X矩阵
 X=X/256
 return W1,W2,b1,b2,X,train_set

def Forward_MAX(W,X,b,train_set,num_of_Pictures,Thresholdvalue):  #max作为Loss Function
    F=W.dot(X)+b  #引入偏置
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
    return {'Loss_array':Loss_array,
            'Loss_total':Loss_total
            }


def Forward_SOFTMAX(W,X,b,train_set,num_of_Pictures):  #softmax作为Loss Function
    F=W.dot(X)+b
    F_origin=F.copy()
    n=num_of_Pictures
    Loss_array=np.zeros([1,n])
    #做一个标准化，避免溢出
    maxvalue=np.amax(F)
    F=F-maxvalue
    #print(F)
    for i in range(n):
        Correct_label=train_set['labels'][i]
        #print(np.exp(F[Correct_label][i]))
        #print(np.sum(np.exp(F[:,i])))
        Loss_array[0][i]=-np.log10(np.exp(F[Correct_label][i])/np.sum(np.exp(F[:,i]))) #运用softmax函数 -log10(exp(correct)/sum(exp(each)))
    Loss_total=np.sum(Loss_array)
    return {'Loss_array':Loss_array,
            'Loss_total':Loss_total,
            'F':F,
            'W':W,
            'b':b,
            'F':F,
            'X':X,
            'F_origin':F_origin
            }

def Backward_MAX(W,X,b,Gradient_Loss_In,Loss_array,train_set,num_of_Pictures,Step_size):  #对用max进行的一层，进行backpropagation
    #Back propagation
    n=num_of_Pictures
    #Step1:接受下一级回传的梯度
    Gradient_Loss_array=np.multiply(Gradient_Loss_In,Loss_array) #下一级计算出的梯度，要乘该级的系数，作为Gradient_F进行计算的来源
    #Step2:求出F矩阵的梯度
    Gradient_F_above_0=np.array((Loss_array>0),float)  #找出Loss_array内元素进行max后取了sj-sr+1的位置
    Gradient_F=np.multiply(Gradient_Loss_array,Gradient_F_above_0.copy())  #考虑下一级回溯过来的矩阵梯度，进行点乘
    for i in range(n):
        Correct_label=train_set['labels'][i] 
        Gradient_F[Correct_label][i]=-1*np.sum(np.multiply(Gradient_Loss_array[:,i],Gradient_F_above_0[:,i]))
    
    #Step3:求出W矩阵的梯度
    Gradient_W=np.zeros([10,3072])
    for i in range(10):
        for j in range(3072):
            Gradient_W[i][j]=np.sum(np.multiply(Gradient_F[i,:],X[j,:]))
            
    #求出X矩阵的梯度 (当loss作为X时，可以使用该梯度继续回溯)
    Gradient_X=np.zeros([3072,n])
    for i in range(3072):
        for j in range(n):
            Gradient_X[i][j]=np.sum(np.multiply(Gradient_F[:,j],W[:,i]))
      
    #求出b的梯度
    Gradient_b=np.sum(Gradient_F)/100000
    
    #Step4:修正W矩阵和b
    #对于训练集为10时 0.00000001
    W=W-Step_size*Gradient_W
    b=b-Step_size*Gradient_b
    
    return {'W_update':W,
            'X':X,
            'b_update':b,
            'Gradient_W':Gradient_W,
            'Gradient_F':Gradient_F,
            'Gradient_b':Gradient_b,
            'Gradient_X':Gradient_X,
            'Gradient_Loss_In':Gradient_Loss_In
            }
    

    
def Backward_SOFTMAX(W,X,b,F,Gradient_Loss_In,Loss_array,train_set,num_of_Pictures,Step_size):
    n=num_of_Pictures
    #Step1:接受下一级回传的梯度
    Gradient_Loss_array=np.multiply(Gradient_Loss_In,Loss_array) #下一级计算出的梯度，要乘该级的系数，作为Gradient_F进行计算的来源
    #Step2:求出F矩阵的梯度
    Gradient_F=np.zeros([10,n])
    log10=np.log(10)
    for i in range(n):
        Correct_label=train_set['labels'][i] 
        sum_of_exp=np.sum(np.exp(F[:,i]))
        Gradient_F[:,i]=np.exp(F[:,i])/(sum_of_exp*np.exp(F[Correct_label][i])*log10)  #非正确label的梯度，省略系数np.log(10)
        Gradient_F[Correct_label][i]=(np.exp(F[Correct_label][i])-sum_of_exp)/(sum_of_exp*log10)   #正确label的梯度
        Gradient_F[:,i]=Gradient_Loss_array[0][i]*Gradient_F[:,i]  #把上一级留下来的系数乘进去
    
    #Step3:求出W矩阵的梯度
    Gradient_W=np.zeros([10,10])
    for i in range(10):
        for j in range(10):
            Gradient_W[i][j]=np.sum(np.multiply(Gradient_F[i,:],X[j,:]))
            
    #求出X矩阵的梯度 (当loss作为X时，可以使用该梯度继续回溯)
    Gradient_X=np.zeros([10,n])
    for i in range(10):
        for j in range(n):
            Gradient_X[i][j]=np.sum(np.multiply(Gradient_F[:,j],W[:,i]))
      
    #求出b的梯度
    Gradient_b=np.sum(Gradient_F)/100000
    
    #Step4:修正W矩阵和b
    #对于训练集为10时 0.00000001
    W=W-Step_size*Gradient_W
    b=b-Step_size*Gradient_b   
        
    return {'W_update':W,
            'X':X,
            'b_update':b,
            'Gradient_W':Gradient_W,
            'Gradient_F':Gradient_F,
            'Gradient_b':Gradient_b,
            'Gradient_X':Gradient_X,
            'Gradient_Loss_In':Gradient_Loss_In
            }
    
def Neural_Network_train(num_of_Pictures,Thresholdvalue,Step_size):
    n=num_of_Pictures
    [W1,W2,b1,b2,X,train_set]=select_training_set(n)  #把测试数据准备好，模型初始化
    [Result_Max,Result_SM,Result_Back_Max,Result_Back_SM]=One_Cycle(W1,W2,X,b1,b2,train_set,num_of_Pictures,Thresholdvalue,Step_size)
    print("Training Loss Layer1=%f Layer2=%f" % (Result_Max['Loss_total'],Result_SM['Loss_total']))      
    for i in range(0):
        [Result_Max,Result_SM,Result_Back_Max,Result_Back_SM]=One_Cycle(Result_Back_Max['W_update'],Result_Back_SM['W_update'],X,Result_Back_Max['b_update'],Result_Back_SM['b_update'],train_set,num_of_Pictures,Thresholdvalue,Step_size)
        print("Training Loss Layer1=%f Layer2=%f" % (Result_Max['Loss_total'],Result_SM['Loss_total']))      
    return Result_Max,Result_SM,Result_Back_Max,Result_Back_SM#,Result_Max2,Result_SM2,Result_Back_Max2,Result_Back_SM2
    

    
def One_Cycle(W1,W2,X,b1,b2,train_set,num_of_Pictures,Thresholdvalue,Step_size):
    n=num_of_Pictures
    Result_Max=Forward_MAX(W1,X,b1,train_set,num_of_Pictures,Thresholdvalue)
    Result_SM=Forward_SOFTMAX(W2,Result_Max['Loss_array'],b2,train_set,num_of_Pictures)
    Result_Back_SM=Backward_SOFTMAX(W2,Result_Max['Loss_array'],b2,Result_SM['F'],np.ones([1,n]),Result_SM['Loss_array'],train_set,num_of_Pictures,Step_size)
    Result_Back_Max=Backward_MAX(W1,X,b1,Result_Back_SM['Gradient_X'],Result_Max['Loss_array'],train_set,num_of_Pictures,Step_size)
    return Result_Max,Result_SM,Result_Back_Max,Result_Back_SM 

#def select_training_set(num_of_Pictures):
#def Forward_MAX(W,X,b,train_set,num_of_Pictures,Thresholdvalue):  
#def Forward_SOFTMAX(W,X,b,train_set,num_of_Pictures):  
#def Backward_SOFTMAX(W,X,b,F,Gradient_Loss_In,Loss_array,train_set,num_of_Pictures,Step_size):
#def Backward_MAX(W,X,b,Gradient_Loss_In,Loss_array,train_set,num_of_Pictures,Step_size): 


#用测试集，测试SVM模型       
def test_Nuural_Network(W1,b1,W2,b2,num_of_testing_Pictures):
    n=num_of_testing_Pictures
    test_set=CIFAR.test_minibatch(n)
    X_test=test_set['data'].T
    X_test=X_test/256
    F1=W1.dot(X_test)+b1
    F2=W2.dot(F1)+b2    
    Labels_Predict=np.argmax(F2,axis=0)    
    Labels_True_list=test_set['labels']
    Labels_True=np.array(Labels_True_list,'int')
    comp=(Labels_Predict==Labels_True)          
    true_num=np.sum(comp)
    percent=true_num/n
    return percent,true_num,n