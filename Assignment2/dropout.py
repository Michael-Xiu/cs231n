import numpy as np
import CIFAR_use
import matplotlib.pyplot as plt

#dropout 前向传递
def  dropout_forward(x,  dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']  #这里的p时keep probability
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':

        ########################################################################
        #  TODO:  Implement  the  training  phase  forward  pass  for  inverted
        #  dropout.
        #  Store  the  dropout  mask  in  the  mask  variable.
        ########################################################################

        mask = (np.random.rand(*x.shape) <= p)/ p #分子为mask的位置,分母p为做inverted dropout所要做的规范化
        out = x * mask  # drop

    elif mode == 'test':  #训练时不做dropout

        out = x

    cache  =  (dropout_param,  mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

#dropout 后项传递
def dropout_backward(dout,cache):
    dropout_param,mask = cache
    mode = dropout_param['mode']

    dx=None

    if mode == 'train':

        dx = dout * mask  #dropout的反向传播矩阵仅仅为dout*掩码矩阵

    elif mode == 'test':

        dx = dout

    return dx


