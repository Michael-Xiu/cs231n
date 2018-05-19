import numpy as np
import CIFAR_use
#from cs231n.data_utils import get_CIFAR10_data #3维图像的取法和DNN中是不同的

#快速层包cs231n.fast_layers：conv_forward_fast conv_backward_fast max_pool_forward_fast max_pool_backward_fast
from cs231n.fast_layers import *
#注：在使用时需要借助Cython来生成C扩展，加快运行速度
#C:\Users\Michael\Desktop\Project\CNN\Sranford CNN\homework\assignment2\cs231n>python setup.py build_ext --inplace

#基础层包cs231n.layers--把内容（基础层定义）写在该CNN文件中，方便学习

#将层整合包cs231n.layer_utils--将内容（多层整合）写在该CNN文件中，方便学习

#整个CNN架构搭建包cs231n.classifiers.cnn



# CNN naive layers：just for learning
''' 
def  conv_forward_naive(x,  w,  b,  conv_param):
    """
    A  naive  implementation  of  the  forward  pass  for  a  convolutional  layer.

    The input consists of N data points, each with  C  channels, height  H and width W.
    We convolve each input with F different filters, where each filter spans all C channels and
    has height HH and width WW.Input:
    ‐    x: Input data of shape(N, C, H, W) 输入x为4维矩阵
    ‐    w: Filter weights of shape(F, C, HH, WW) 输入w为4维矩阵 filter的深度C必须与数据的深度相同
    ‐    b: Biases, of shape(F, )
    ‐    conv_param: A dictionary with  the  following  keys:
        ‐    'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
        ‐    'pad': The number of pixels that will be used to zero‐pad the input.

    Returns a tuple of:
    ‐    out: Output data, of shape(N, F, H',  W')  where H'  and  W' are given by
         H_out  =  1  +  (H  +  2  *  pad  ‐  HH)  /  stride W_out = 1 + (W + 2 * pad  ‐  WW) / stride
    ‐    cache: (x, w, b, conv_param)
    """

    out = None

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    stride, pad = conv_param['stride'], conv_param['pad'] #把步长stride和zero-padding的位宽大小从变量字典中取出来

    H_out = 1 + (H + 2 * pad - HH) / stride   #根据数据矩阵大小，以及filter矩阵大小、步长和pad大小，计算出经过conv层之后的输出矩阵大小
    W_out  =  1  +  (W  +  2  *  pad  - WW)  /  stride

    H_out = int(H_out)  #由于上方计算结果为float，转为int
    W_out = int(W_out)
    out = np.zeros((N, F, H_out, W_out))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    #################################################################################################################
    # pad（array，pad_width，mode，**kwars）

    # 其中array为要填补的数组（input）
    # pad_width是在各维度的各个方向上想要填补的长度,如（（2，3），（4，5）），如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
    # mode为填补类型，即怎样去填补，有“constant”，“edge”等模式，如果为constant模式，就得指定填补的值。
    # 在本例中，进行zero-padding后，N不受影响，C不受影响，H上下扩展pad位0，W左右扩展pad位0
    ##################################################################################################################

    #求out[:,K,i,j]  :可以认为out矩阵具有N个(1st)相似的三维矩阵F(2nd)*H_out(3rd)*W_out(4th)，成为单个样本经conv输出矩阵out1
    # 1、先考虑conv的适用数据对象
    for i in range(H_out):
        for j in range(W_out):  #对于N个out1，其i,j位置对应的是N的样本相同位置的数据体，用x_pad_cover表示N个out1其i,j位置所使用的N个三维数据体(4维)
            x_pad_cover = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] #out矩阵每一个元素体对应N个数据体x_pad同一位置的一小块cover部分，立体

            # 2、 再考虑输出矩阵的值
            for k in range (F):  #考虑输出矩阵不同层，对i,j位置来说，其差别在于同一个数据体使用了不同的filter来计算
                for m in range(N):  #考虑不同样本下，x_pad_cover去不同样本的数据体，和第K个filter进行点乘，三维矩阵*三维矩阵，结果再求和
                     out[m,k,i,j] = np.sum(x_pad_cover[m, :, :, :] * w[k, :, :, :],axis=(0, 1, 2))  #out矩阵所有N个输出，对于第K层而言，其元素值都是每一个元素对应的数据体去点乘第K个filter，然后进行求和

    out = out + b[None, :, None, None] # 对于每一个filter加上一个偏置b，即对out矩阵每一层加上独立的偏置
                                       # b为一维向量，通过[None,:,None,None]扩展为4维矩阵，其中第二维为向量值，可以看作只对out矩阵第二维(F)进行了不同的加和

    cache = (x,w,b,conv_param)
    return out,cache



def  conv_backward_naive(dout,  cache):
    """
    A  naive  implementation  of  the  backward  pass  for  a  convolutional layer.
    Inputs:
    ‐	dout:  Upstream  derivatives.
    ‐	cache:  A  tuple  of  (x,  w,  b,  conv_param)  as  in  conv_forward_naive Returns a tuple of:
    ‐	dx:  Gradient  with  respect  to  x
    ‐	dw:  Gradient  with  respect  to  w
    ‐	db:  Gradient  with  respect  to  b
    """

    dx,  dw,  db  =  None,  None,  None

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad  - WW) / stride
    H_out = int(H_out)  # 由于上方计算结果为float，转为int
    W_out = int(W_out)

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # 1. db - 偏置向量的梯度
    db = np.sum(dout, axis=(0,2,3))  # db即对dout矩阵每一层进行求和即可，因为每一层对应的就是同一个filter

    #[循环中注意随着循环变量的增多，所要运算量是不断减少的，要排除干扰]
    for i in range(H_out):
        for j in range(W_out):
            x_pad_cover = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]   #(i,j)对应的数据块 (4D)


            # 2. 计算不同filter对应的dw
            for k in range (F):
                dw[k, :, :, :] += np.sum(x_pad_cover * (dout[:, k, i, j])[:, None, None, None], axis=0)
                #将N个样本dout(K,i,j)对应的值，映射到N个样本(i,j)对应的数据体，各自进行点乘，随后相应位置加和

            # 3. 计算不同样本下的dx_pad
            for m in range(N):  #对应第m个样本的数据块(3D)和F个filter进行了卷积操作
                dx_pad[m, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((w[:, :, :, :] * (dout[m, :, i, j])[:, None, None, None]),axis=0)
                #dout[m, :, i, j])[:, None, None, None]是把dout中(i,j)对应的第二维数据抽出，并作为第一维数据扩展为四维矩阵，目的是让每个filter对应上dout中(i,j)的每个梯度值
                #将第N个样本dout中 K个(i,j)值，映射到K个filter上，各自进行点乘，随后相应位置加和，将会得到第N个样本dout(i,j)所对应的第N个样本的数据体的梯度

    ####################################################################################################################
    #简述计算conv层梯度dw,dx_pad的步骤
    # 1. 从dout中确定(i,j)位置，找到(i,j)位置对应各个样本的数据体
    #   2. 计算dw时，分开不同的filter来计算，不同的filter对应dout的不同层
    #      计算第K个filter的梯度值，则需要把dout中第K层抽出来，同时确认不同样本(i,j)下对应的值，下一步将这些值映射到N个样本，去点乘
    #      各自的(i,j)数据体，就可以求和出第K个filter在dout(i,j)下的梯度。要计算第K个filter总梯度，则需要遍历(i,j)平面进行求和
    #   3. 计算dx_pad时，分开不同的样本进行计算，不同的样本对应dout不同长方体
    #      计算第m个样本的梯度值，则需要把dout中第m个长方体中(i,j)所对应的深度为F的向量取出，将其映射到F个filter，去点乘各个filter，
    #      就可以求和出第m个样本dout(i,j)对应的数据体的梯度。
    #注意：在进行filter的shift时，这里所说的数据体会有重叠，所以这里的总梯度不是科学的说法，实际上最终的梯度都是不断在各个来源加和
    #     的结果，
    ####################################################################################################################

    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db



def  max_pool_forward_naive(x,  pool_param):
    """
    A  naive  implementation  of  the  forward  pass  for  a  max  pooling  layer. Inputs:
    ‐	x:  Input  data,  of  shape  (N,  C,  H,  W)
    ‐	pool_param:  dictionary  with  the  following  keys:
    ‐	'pool_height':  The  height  of  each  pooling  region
    ‐	'pool_width':  The  width  of  each  pooling  region
    ‐	'stride':  The  distance  between  adjacent  pooling  regions Returns a tuple of:
    ‐	out: Output data
    ‐	cache:  (x,  pool_param)
    """
    out = None

    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    H_out = 1 + (H - HH) / stride
    W_out = 1 + (W - WW) / stride
    H_out = int(H_out)  # 由于上方计算结果为float，转为int
    W_out = int(W_out)


    out = np.zeros((N,C,H_out,W_out))

    for i in range(H_out):
        for j in range(W_out):
            x_cover = x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] #x_cover 为N个out的(i,j)位置 所对应N个样本的数据体
            out[:, :, i, j] = np.max(x_cover, axis=(2,3))  #max_pooling，对数据体的(i,j)即第三第四维求和


    cache = (x, pool_param)
    return out, cache



def  max_pool_backward_naive(dout,  cache):
    """
    A  naive  implementation  of  the  backward  pass  for  a  max  pooling  layer. Inputs:
    ‐	dout:  Upstream  derivatives
    ‐	cache:  A  tuple  of  (x,  pool_param)  as  in  the  forward  pass. Returns:
    ‐	dx:  Gradient  with  respect  to  x
    """

    (x, pool_param) = cache
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    H_out = 1 + (H - HH) / stride
    W_out = 1 + (W - WW) / stride
    H_out = int(H_out)  # 由于上方计算结果为float，转为int
    W_out = int(W_out)


    dx = np.zeros_like(x)

    # pooling层梯度为：dout的某个值（i，j），这个值就是对应pooling对象中max位置的梯度
    for i in range(H_out):
        for j in range(W_out):
            x_cover = x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] #x_cover 为N个out的(i,j)位置 所对应N个样本的数据体
            max_x_cover = np.max(x_cover, axis=(2,3))  #max_pooling，对数据体的(i,j)即第三第四维求和,max_x_cover为2D
            temp_binary_mask = (x_cover == max_x_cover[:, :, None, None]) #将输出的最大值在（i,j）平面扩展开来，形成和输入一样大小的四维矩阵，匹配最大值位置
            dx[:, :, i*stride: i*stride + HH, j*stride: j*stride+WW] += temp_binary_mask * (dout[:, :, i, j])[:, :, None, None]  #这一步和上一部类似

    return dx

    ####################################################################################################################
    #简述计算pooling层梯度dx的步骤  （首先明确pooling只有一个，而不像conv有几个filters）
    # 1. 从dout中确定(i,j)位置，找到(i,j)位置对应各个样本的数据体
    #   2. 按着forward的方式，求出该数据体经过max pooling层之后的输出，二维矩阵第一维为哪个样本，第二维为哪层数据
    #      将这个二维输出扩展为四维，原理是同一个样本同一层的数据是相同的，然后将这个矩阵和数据体进行比对，则可以确定出究竟数据体
    #      中哪些数据是max。这些max数据对应的梯度就是1*dout对应值，其余梯度为0
    ####################################################################################################################
'''

# 基础层：大部分从FNN中直接复制
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = x * (x >= 0)

    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    dx = dout * (x >= 0)

    return dx

def affine_forward(x,W,b):  #affine仿射->从2D坐标到其他2D坐标的线性映射:平移 (translation)、缩放 (scale)、翻转 (flip)、旋转 (rotation) 和错切 (shear)
    #x_rsp中会将x转为（N,D）,w:(D,M), b:(M,), out(N,M), cache(x,w,b)_
    out=None
    N=x.shape[0] #通过行数确定x中样本的数量N
    x_rsp=x.reshape(N,-1) #每一个图像转成一维
    out=x_rsp.dot(W)+b
    cache=(x,W,b)

    return out,cache

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

def softmax_loss(x,y):
    probs=np.exp(x-np.max(x,axis=1,keepdims=True))  #对每一个特征规范化，避免exp溢出
    probs/=np.sum(probs,axis=1,keepdims=True)  #对每一个数据都求softmax
    N=x.shape[0]
    loss=-np.sum(np.log(probs[np.arange(N),y]))/N  #抽取正确label对应位置，加和平均作为loss
    dx=probs.copy()
    dx[np.arange(N),y]-=1  #softmax梯度：非正确label值就等于softmax值，正确label位置还要额外-1
    dx/=N
    return loss,dx

#整合层
def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
      其中conv_param主要包括conv_param['stride'], conv_param['pad']

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
      其中conv_param包括conv_param['stride'], conv_param['pad']
      其中pool_param包括pool_param['pool_height'], pool_param['pool_width']， pool_param['stride']

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


#三层卷积神经网络 from cs231n.classifiers.cnn
#结构为：conv - relu - 2x2 max pool - affine - relu - affine - softmax  卷积层-全连接层-全连接层
class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        C, H, W = input_dim

        # 卷积层作为W1，为四维矩阵；其偏置作为b1，为一维向量
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        # Fully-connected层1：在本例中，Padding and stride chosen to preserve the input spatial size
        self.params['W2'] = weight_scale * np.random.randn(int(H / 2) * int(W / 2) * num_filters, hidden_dim)  # 与卷积层的输出尺寸有关
        self.params['b2'] = np.zeros(hidden_dim)

        # Fully-connected层2
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']


        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]

        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################

        conv_forward_out_1, cache_forward_1 = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        affine_forward_out_2, cache_forward_2 = affine_forward(conv_forward_out_1, self.params['W2'], self.params['b2'])
        affine_relu_2, cache_relu_2 = relu_forward(affine_forward_out_2)
        scores, cache_forward_3 = affine_forward(affine_relu_2, self.params['W3'], self.params['b3'])

        if y is None:

            return scores

        loss = 0
        loss, dout = softmax_loss(scores, y)  #该函数：1.计算scores经过softmax后的loss矩阵 2.计算scores矩阵相对于loss的梯度

        # 采用L2正则化项
        loss += self.reg * 0.5 * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2))
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        grads = {}

        dX3, grads['W3'], grads['b3'] = affine_backward(dout, cache_forward_3)
        dX2 = relu_backward(dX3, cache_relu_2)
        dX2, grads['W2'], grads['b2'] = affine_backward(dX2, cache_forward_2)
        dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)

        grads['W3'] = grads['W3'] + self.reg * self.params['W3']
        grads['W2'] = grads['W2'] + self.reg * self.params['W2']
        grads['W1'] = grads['W1'] + self.reg * self.params['W1']


        return loss, grads
