import numpy as np
import CIFAR_use
import matplotlib.pyplot as plt
import BN


# class FNN: #Fully connected neural network
# 测试代码为 t_FNN.py

'''--------------------------------基础层构建----------------------------------------------'''
#affine_foward为线性层的前向函数
def affine_forward(x,W,b):  #affine仿射->从2D坐标到其他2D坐标的线性映射:平移 (translation)、缩放 (scale)、翻转 (flip)、旋转 (rotation) 和错切 (shear)
    #x_rsp中会将x转为（N,D）,w:(D,M), b:(M,), out(N,M), cache(x,w,b)_
    out=None
    N=x.shape[0] #通过行数确定x中样本的数量N
    x_rsp=x.reshape(N,-1) #每一个图像转成一维
    out=x_rsp.dot(W)+b
    cache=(x,W,b)

    return out,cache

#affine_backward为线性的反向传播函数
def affine_backward(dout,cache):
    #dout:(N,M), x:(N,D), w:(D,M),dx
    x,W,b=cache
    dx,dw,db=None,None,None
    N=x.shape[0] #通过行数确定x中样本的数量N
    x_rsp=x.reshape(N,-1) #每一个图像转成一维
    dx=dout.dot(W.T)    #对于X*W=Out dx=dOut*W' dw=x'*dOut
    dx=dx.reshape(*x.shape)
    dw=x_rsp.T.dot(dout)
    db=np.sum(dout,axis=0)

    return dx,dw,db

#ReLu层前向传递
def relu_forward(x):
    out=x*(x>=0)
    cache=x
    return out,cache

#ReLu层后向传递
def relu_backward(dout,cache):
    x=cache
    dx=(x>=0)*dout
    return dx

#Batch Normalization 前向传递
def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)  #如果有输入eps则用输入值，否则用初始化值1e-5
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None

    if mode == 'train':  #运用Batch Normalization进行训练
        sample_mean = np.mean(x, axis=0)  #1. mini-batch mean
        sample_var = np.var(x, axis=0)    #2. mini-batch variance
        x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps))  #3. normalize
        out = gamma * x_hat + beta  #4. scale and shift 这里gamma，beta是一个行向量，目的是对一个特征进行scale and shift
        cache = (gamma, x, sample_mean, sample_var, eps, x_hat)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean  #这两步是SGD+Momentum算法的运用
        running_var = momentum * running_var + (1 - momentum) * sample_var

        #############################################################################
        #  TODO: Implement the training-time forward pass for batch normalization.   #
        #  Use minibatch statistics to compute the mean and variance, use these      #
        #  statistics to normalize the incoming data, and scale and shift the        #
        #  normalized data using gamma and beta.                                     #
        #                                                                           #
        #  You should store the output in the variable out. Any intermediates that   #
        #  you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        #  You should also use your computed sample mean and variance together with  #
        #  the momentum variable to update the running mean and running variance,    #
        #  storing your result in the running_mean and running_var variables.        #
        # ##############################################################################

    elif mode == 'test':  #运用Batch Normalization进行测试
        # scale = gamma / (np.sqrt(running_var + eps))
        # out = x * scale + (beta - running_mean * scale)

        #对输入数据进行相同的规范化，将上两句改为下两句感觉会更加清晰
        #测试时运用的是running_mean 和running_variance，是训练过程中积累下来的均值方差
        x_hat = (x - running_mean)/ (np.sqrt(running_var + eps))
        out = gamma * x_hat + beta

        #############################################################################
        #  TODO: Implement the test-time forward pass for batch normalization. Use   #
        #  the running mean and variance to normalize the incoming data, then scale  #
        #  and shift the normalized data using gamma and beta. Store the result in   #
        #  the out variable.                                                         #
        # #############################################################################

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache

#Batch Normalization 后向传递（computational graphs）
def batchnorm_backward(dout, cache):
    #############################################################################
    #  forward参考的是Will的代码，cache取出数据采用和forward匹配的方式               #
    #  backward参考kratzert代码                                                   #
    #  每一步计算的方式可以参考我自己画的图解                                        #
    # 在forward只用了4步最直接的方式求出out，backpropagate时我们把每一步的细节divide  #
    # and conquer,目的是用最基本简单操作构成一幅computational graphs,方便计算梯度    #
    # 所以cache取出数据时需要补充计算一些中间步骤的结果
    # #############################################################################

    dx, dgamma, dbeta = None, None, None

    # unfold the variables stored in cache
    gamma, x, sample_mean, sample_var, eps, x_hat = cache

    #补充计算
    xmu = x - sample_mean
    var = sample_var
    sqrtvar = np.sqrt(var + eps)
    ivar = 1./ sqrtvar

    # get the dimensions of the input/output
    N, D = dout.shape

    # step9
    dbeta = np.sum(dout, axis=0)
    dgammax = dout  # not necessary, but more understandable

    # step8
    dgamma = np.sum(dgammax * x_hat, axis=0)
    dxhat = dgammax * gamma

    # step7
    divar = np.sum(dxhat * xmu, axis=0)
    dxmu1 = dxhat * ivar

    # step6
    dsqrtvar = -1. / (sqrtvar ** 2) * divar

    # step5
    dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar

    # step4
    dsq = 1. / N * np.ones((N, D)) * dvar

    # step3
    dxmu2 = 2 * xmu * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

    # step1
    dx2 = 1. / N * np.ones((N, D)) * dmu

    # step0
    dx = dx1 + dx2

    return dx, dgamma, dbeta
    #知乎源代码：思路不清晰，不用理会
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the #
    # results in the dx, dgamma, and dbeta variables.                            #
    #############################################################################
    # gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
    # #(gamma, x, sample_mean, sample_var, eps, x_hat)
    #
    # N = x.shape[0]
    #
    # dx_1 = gamma * dout   # step9+8: dx
    # dx_2_b = np.sum((x - u_b) * dx_1, axis=0)  # step7: divar (there xμ=x-u_b here)
    # dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1   # step5
    # dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b
    # dx_4_b = dx_3_b * 1
    # dx_5_b = np.ones_like(x) / N * dx_4_b
    # dx_6_b = 2 * (x - u_b) * dx_5_b
    # dx_7_a = dx_6_b * 1 + dx_2_a * 1
    # dx_7_b = dx_6_b * 1 + dx_2_a * 1
    # dx_8_b = -1 * np.sum(dx_7_b, axis=0)
    # dx_9_b = np.ones_like(x) / N * dx_8_b
    # dx_10 = dx_9_b + dx_7_a
    #
    # dgamma = np.sum(x_hat * dout, axis=0)
    # dbeta = np.sum(dout, axis=0)
    # dx = dx_10

#Batch Normalization 后向传递加速（表达式）
def batchnorm_backward_alt(dout, cache):
    # Backward的加速形式：即每个梯度都是一条式子计算出来，不再使用computational graphs
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the #
    # results in the dx, dgamma, and dbeta variables.                            #
    #                                                                            #
    # After computing the gradient with respect to the centered inputs, you      #
    # should be able to compute gradients with respect to the inputs in a        #
    # single statement; our implementation fits on a single 80-character line.   #
    #############################################################################
    gamma, x, sample_mean, sample_var, eps, x_hat = cache
    N = x.shape[0]
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - sample_mean) * -0.5 * np.power(sample_var + eps, -1.5), axis=0)
    dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)
    dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dvar * 2.0 / N * (x - sample_mean) + 1.0 / N * dmean
    dgamma = np.sum(x_hat * dout, axis=0)

    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta

#affine+relu 前向传递
def affine_relu_forward(x,W,b):
    a,fc_cache=affine_forward(x,W,b)
    out,relu_cache=relu_forward(a)
    cache=(fc_cache,relu_cache)
    return out,cache  #返回值out是Relu层输出，而cache则包括映射层和Relu层两个cache

#affine+relu 后向传递
def affine_relu_backward(dout,cache):
    fc_cache,relu_cache=cache
    da=relu_backward(dout,relu_cache)
    dx,dw,db=affine_backward(da,fc_cache)
    return dx,dw,db

#affine+bn+relu 前向传递
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

#affine+bn+relu 后向传递
def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward_alt(dbn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


#SVM损失函数；既求输出，也求梯度
def svm_loss(x,y):
    #x(N,C) x[i,j]表示第i个输入是第j个的分数  y(N,),x[i]的标签
    #不正确label：指的是 其减去正确label值+阈值>0的位置

    N=x.shape[0] #通过行数确定x中样本的数量N
    correct_class_scores=x[np.arange(N),y]  #一个图像预测对应一行数据，把对应正确label位置的值取出来，组成一个数组
    margins=np.maximum(0,x-correct_class_scores[:,np.newaxis]+1.0)  #newaxis为增加一个轴，求出loss矩阵
    margins[np.arange(N),y]=0  #计算SVM loss时，正确位置不参与计算，所以其loss设为0
    loss=np.sum(margins)/N

    #SVM梯度：非正确label位置，如果loss>0，则其=1,同时正确label位置的梯度值会+1
    dx=np.zeros_like(x)  #生成一个和x矩阵同样大小的0矩阵
    dx[margins>0]=1      #不正确的位置，由于loss是加和，所以为1（正确位置的loss=0）
    num_pos=np.sum(margins>0,axis=1)  #把loss矩阵每行中>0的位置个数加起来
    dx[np.arange(N),y]-=num_pos  #正确的位置，有多少个不正确label，其梯度就等于多少
    dx/=N  #由于求loss时/N，所以整体/N
    return loss,dx  #dx指的是进入svm层计算loss前那个得分矩阵的梯度，相当于dOut(Out=x*W)

#softmax损失函数；既求输出，也求梯度
def softmax_loss(x,y):
    probs=np.exp(x-np.max(x,axis=1,keepdims=True))  #对每一个特征规范化，避免exp溢出
    probs/=np.sum(probs,axis=1,keepdims=True)  #对每一个数据都求softmax
    N=x.shape[0]
    loss=-np.sum(np.log(probs[np.arange(N),y]))/N  #抽取正确label对应位置，加和平均作为loss
    dx=probs.copy()
    dx[np.arange(N),y]-=1  #softmax梯度：非正确label值就等于softmax值，正确label位置还要额外-1
    dx/=N
    return loss,dx
'''--------------------------------基础层构建end---------------------------------------------'''



'''两层神经网络'''
class TwoLayerNet(object):
    def __init__(self,input_dim=3*32*32,hidden_dim=100,num_classes=10,weight_scale=1e-3,reg=0.0):
        #input_dim指input得维度，图像为3072，hidden_dim指隐藏层神经元个数(特征数),num_classes指类的数量，weigh_scale则是给外部接口，使得可以改动初始系数矩阵W的大小，reg为正则项
        self.params={}
        self.reg=reg
        self.params['W1'] = weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self,X,y=None):
        scores=None
        ar1_out,ar1_cache=affine_relu_forward(X,self.params['W1'],self.params['b1']) #第一层带有relu
        a2_out,a2_cache=affine_forward(ar1_out,self.params['W2'],self.params['b2'])  #第二层带有softmax_loss
        scores=a2_out  #scores是评分矩阵，该矩阵还要求softmax才能得到softmax loss

        if y is None:
            return scores

        loss,grads=0,{}
        loss,dscores=softmax_loss(scores,y)  #返回loss和该层的梯度矩阵
        #使用时，正则项为0.5乘以矩阵每个元素的平方和
        loss=loss+0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])


        dx2,dw2,db2=affine_backward(dscores,a2_cache)  #对应第二层的反向
        grads['W2']=dw2+self.reg*self.params['W2']  #求loss还加上了正则项，此正则项直接与W1，W2相关，所以正则项的backpropagate直接指向W1，W2
        grads['b2']=db2

        dx1,dw1,db1=affine_relu_backward(dx2,ar1_cache)  #对应第一层的反向
        grads['W1']=dw1+self.reg*self.params['W1']
        grads['b1']=db1

        return loss,grads

    ''' 由于上述代码需要经过solver.py进行优化，内部输入输出变量设置需要和solver匹配'''
    '''------------------------------------------end----------------------------------------------------------------'''

    '''--------------------------------三层神经网络（在两层基础上修改）----------------------------------------------'''

'''三层神经网络（两层基础上改）'''
class ThreeLayerNet(object):
    def __init__(self, input_dim=3 * 32 * 32, hidden_dim1=100,hidden_dim2=100,num_classes=10, weight_scale=1e-3, reg=0.0):
        # input_dim指input得维度，图像为3072，hidden_dim指隐藏层神经元个数(特征数),num_classes指类的数量，weigh_scale则是给外部接口，使得可以改动初始系数矩阵W的大小，reg为正则项
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim1)
        self.params['b1'] = np.zeros(hidden_dim1)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim1, hidden_dim2)
        self.params['b2'] = np.zeros(hidden_dim2)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim2, num_classes)
        self.params['b3'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        scores = None
        ar1_out, ar1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])  # 第一层带有relu
        ar2_out, ar2_cache = affine_relu_forward(ar1_out, self.params['W2'], self.params['b2'])  # 第二层带有relu
        a3_out, a3_cache = affine_forward(ar2_out, self.params['W3'], self.params['b3'])  # 第三层带有softmax_loss
        scores = a3_out  # scores是评分矩阵，该矩阵还要求softmax才能得到softmax loss

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)  # 返回loss和该层的梯度矩阵
        # 使用时，正则项为0.5乘以矩阵每个元素的平方和
        loss = loss + 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(
            self.params['W2'] * self.params['W2'])+0.5 * self.reg * np.sum(self.params['W3'] * self.params['W3'])

        dx3, dw3, db3 = affine_backward(dscores, a3_cache)  # 对应第三层的反向
        grads['W3'] = dw3 + self.reg * self.params['W3']  # 求loss还加上了正则项，此正则项直接与W1，W2相关，所以正则项的backpropagate直接指向W1，W2
        grads['b3'] = db3

        dx2, dw2, db2 = affine_relu_backward(dx3, ar2_cache)  # 对应第一层的反向
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2

        dx1, dw1, db1 = affine_relu_backward(dx2, ar1_cache)  # 对应第一层的反向
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1

        return loss, grads
    '''------------------------------------------end----------------------------------------------------------------'''






'''---------------------------数据提取与可视化----------------------------------'''
#数据处理：将CIFAR10图片转成数组，并进行归一化处理（减去均值）
def get_cifar_data(num_training=49000,num_validation=1000,num_test=1000):
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


# loss和accura可视化
def visualize_loss(stats):  # 将train返回的stats进行可视化
    plt.subplot(211)
    plt.plot(stats['loss_history'])
    plt.title('loss history')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(212)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
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
