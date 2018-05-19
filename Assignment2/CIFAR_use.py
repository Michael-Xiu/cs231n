import  pickle
import numpy as np
import os

'''
This code is copied from https://zhuanlan.zhihu.com/p/28204173 
'''

def load_cifar_batch(filename):
    with open(filename,'rb') as f :
        datadict=pickle.load(f,encoding='bytes')
        x=datadict[b'data']
        y=datadict[b'labels']
        x=x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        y=np.array(y)
        return x,y

def load_cifar10(root):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(root,'data_batch_%d' % (b,))
        x,y=load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
    Xtrain=np.concatenate(xs) #1
    Ytrain=np.concatenate(ys)
    del x ,y
    Xtest,Ytest=load_cifar_batch(os.path.join(root,'test_batch')) #2
    return Xtrain,Ytrain,Xtest,Ytest
#Xtrain对应(50000, 32, 32, 3)训练数据集
#Ytrain对应(50000,)训练标签集
#Xtest对应(10000, 32, 32, 3)测试数据集
#Ytest对应(10000,)测试标签集

'''
某些操作：
1）为了加快我们的训练速度，我们只选取5000张训练集，500张测试集
num_training=5000
mask=range(num_training)
x_train=x_train[mask]
y_train=y_train[mask]
num_test=500
mask=range(num_test)
x_test=x_test[mask]
y_test=y_test[mask]

2）数据载入部分已经算是完成了，但是为了欧氏距离的计算，我们把得到的图像数据拉长成行向量
x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
print(x_train.shape,x_test.shape)
'''

