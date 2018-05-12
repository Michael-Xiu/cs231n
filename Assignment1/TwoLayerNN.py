import numpy as np
import CIFAR_use
import matplotlib.pyplot as plt


class Neural_Network:

    def __init__(self,input_size,hidden_size,output_size,std=1e-4): #构造函数

        self.params={}
        self.params['W1']=std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)  #行向量,矩阵和其相加时将复制为同样行数的矩阵；b是行向量而不是列向量，只有这样X中一行的pics对应乘以W中的一列时，加上b中的一个数形成对一类的预测；如果为列向量，则不同类的预测时所加的偏置相同，失去意义
        self.params['W2'] = std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
        '''
        测试时，系数分别为3072,50,10
        W1(3072,50)
        b1(1,50)
        W2(50,10)
        b2(1,10)
        '''


    def loss(self,X,y=None,reg=0.0):

        W1,b1=self.params['W1'],self.params['b1']
        W2,b2=self.params['W2'],self.params['b2']
        N,D=X.shape  #N代表图像数量

        scores=None

        h_output=np.maximum(0,X.dot(W1)+b1) #第一层输出,Relu激活函数
        scores=h_output.dot(W2)+b2  #第二层激活函数前的输出

        if y is None:
            return scores
        loss=None

        shift_scores=scores-np.max(scores,axis=1).reshape((-1,1))  #对于scores矩阵，进行了归一化处理（减去每行的max值）,reshape((-1,1))列矩阵
        softmax_output=np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)  #shift_scores矩阵经过softmax后的矩阵
        loss=-np.sum(np.log(softmax_output[range(N),list(y)]))  #loss是对所有正确label的对数求和,再去-
        loss/=N
        loss+=0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2)) #正则项

        grads={}

        #第二层梯度计算
        dscores=softmax_output.copy()  #softmax矩阵，除正确label位置外，其余梯度为原值
        dscores[range(N),list(y)]-=1   #正确label位置，梯度为原值-1
        dscores/=N
        grads['W2']=h_output.T.dot(dscores)+reg*W2  #F=XW  W'=Xt*Y' X'=Y'*Wt ;正则项导数从loss可以推出
        grads['b2']=np.sum(dscores,axis=0)  #列求和

        #第一层梯度计算
        dh=dscores.dot(W2.T)
        dh_ReLu=(h_output>0)*dh  #ReLu的梯度是矩阵结果大于0处偏导为1，相应乘上反向传来的梯度
        grads['W1']=X.T.dot(dh_ReLu)+reg*W1
        grads['b1']=np.sum(dh_ReLu,axis=0)

        return loss,grads



    #训练模型
    def train(self,X,y,X_val,y_val,learning_rate=1e-3,learning_rate_decay=0.95,reg=1e-5,num_iters=100
              ,batch_size=200,verbose=False):

        num_train=X.shape[0]  #测试时为X(49000, 3072)
        iterations_per_epoch=max(num_train/batch_size,1)  #epoch全数据集，iterations_per_epoch为训练一次全数据集所用的次数 batch_size是每次训练的minibatch，测试时为245
        loss_history=[]  #平均损失函数
        train_acc_history=[]  #平均训练准确率
        val_acc_history=[]  #平均验证准确率

        for it in range(num_iters):  #总共训练100次模型，测试时为1000
            X_batch=None #None相当于初始化
            y_batch=None

            idx=np.random.choice(num_train,batch_size,replace=True) #从49000个训练集内，每次随机抽取200为单位的minibatch
            X_batch=X[idx]   #X_batch取X中被抽作minibatch的序数的行，每一行为一个图像的3072pics
            y_batch=y[idx]
            loss,grads=self.loss(X_batch,y=y_batch,reg=reg)
            loss_history.append(loss)

            #参数更新
            self.params['W1']-=learning_rate*grads['W1']
            self.params['W2']-=learning_rate*grads['W2']
            self.params['b1']-=learning_rate*grads['b1']
            self.params['b2']-=learning_rate*grads['b2']

            if verbose and it%100==0:  #每迭代100次，打印 ;verbose来控制是否在屏幕输出信息
                print('iteration %d / %d : loss %f' % (it,num_iters,loss))

            if it % iterations_per_epoch==0:  #认为是训练数量上和选择的训练集相同时进行一次输出
                #比如：训练模型用49000训练集，选了200为minibatch的大小，那么模型训练245次，每次都会更新minibatch，从而可以认为245*200≈49000
                train_acc=(self.predict(X_batch)==y_batch).mean()  #计算出平均训练准确率
                val_acc=(self.predict(X_val)==y_val).mean()   #计算平均验证准确率
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                #更新学习率
                learning_rate*=learning_rate_decay

        return{ 'loss_history':loss_history,
                'train_acc_history':train_acc_history,
                'val_acc_history':val_acc_history    }


    #预测模型
    def predict(self,X):  #self自动传入
        y_pred=None
        h=np.maximum(0,X.dot(self.params['W1'])+self.params['b1'])
        scores=h.dot(self.params['W2'])+self.params['b2']
        y_pred=np.argmax(scores,axis=1) #scores数组每行最大值为该行对应图像的预测label值
        return y_pred



    #数据处理：将CIFAR10图片转成数组，并进行归一化处理（减去均值）
    def get_cifar_data(self,num_training=49000,num_validation=1000,num_test=1000):
        cifar10_dir='C:/Users/Michael/Desktop/Project/CNN/Sranford CNN/homework/CIFAR-10'
        X_train, y_train, X_test, y_test = CIFAR_use.load_cifar10(cifar10_dir)
        #验证集
        mask=range(num_training,num_training+num_validation)  #实际上取了1000个数据
        X_val=X_train[mask]
        y_val=y_train[mask]
        #训练集
        mask=range(num_training)  #取了49000个数据
        X_train=X_train[mask]
        y_train=y_train[mask]
        #测试集
        mask=range(num_test)
        X_test=X_test[mask]
        y_test=y_test[mask]

        #进行归一化处理（减去均值）
        mean_image=np.mean(X_train,axis=0)
        X_train-=mean_image
        X_val-=mean_image
        X_test-=mean_image

        #每一个图像转成一维
        X_train=X_train.reshape(num_training,-1)
        X_val=X_val.reshape(num_validation,-1)
        X_test=X_test.reshape(num_test,-1)

        return X_train,y_train,X_val,y_val,X_test,y_test
    '''
    默认：training data shape: (49000, 3072)
    training labels shape: (49000,)
    validation data shape: (1000, 3072)
    validation labels shape: (1000,)
    test data shape: (1000, 3072)
    test labels shape: (1000,)
    '''


    #loss和accura可视化
    def visualize_loss(self,stats): #将train返回的stats进行可视化
        plt.subplot(211)
        plt.plot(stats['loss_history'])
        plt.title('loss history')
        plt.xlabel('iteration')
        plt.ylabel('loss')

        plt.subplot(212)
        plt.plot(stats['train_acc_history'],label='train')
        plt.plot(stats['val_acc_history'],label='val')
        plt.title('classification accuracy history')
        plt.xlabel('epoch')
        plt.ylabel('classification accuracy')
        plt.show()


from cs231n.vis_utils import visualize_grid
# Visualize the weights of the network  这里专门安装了future库给vis_utils使用
def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

