import numpy as np
import CIFAR_use
import matplotlib.pyplot as plt

#测试代码为 t_bn.py

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

#Backward的加速形式：即每个梯度都是一条式子计算出来，不再使用computational graphs
def batchnorm_backward_alt(dout, cache):
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

