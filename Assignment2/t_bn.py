import numpy as np
import CIFAR_use
import matplotlib.pyplot as plt
from BN import *


#这个包是官方指南里面漏的，注意要补上
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#这个函数是官方指南里面漏的，注意要补上




'''--------------训练过程测试----------------------------'''
# N, D1, D2, D3 = 200, 50, 60, 3
# X = np.random.randn(N, D1)
# W1 = np.random.randn(D1, D2)
# W2 = np.random.randn(D2, D3)
# a = np.maximum(0, X.dot(W1)).dot(W2)
#
# print ('Before batch normalization:')
# print ('  means: ', a.mean(axis=0))
# print ('  stds: ', a.std(axis=0))
#
# # Means should be close to zero and stds close to one
# print ('After batch normalization (gamma=1, beta=0)')
# a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
# print ('  mean: ', a_norm.mean(axis=0))
# print ('  std: ', a_norm.std(axis=0))
#
# # Now means should be close to beta and stds close to gamma
# gamma = np.asarray([1.0, 2.0, 3.0])
# beta = np.asarray([11.0, 12.0, 13.0])
# a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
# print ('After batch normalization (nontrivial gamma, beta)')
# print ('  means: ', a_norm.mean(axis=0))
# print ('  stds: ', a_norm.std(axis=0))


'''--------------测试过程测试----------------------------'''
# N,D1,D2,D3=200,50,60,3
# W1 = np.random.randn(D1, D2)
# W2 = np.random.randn(D2, D3)
# bn_param = {'mode': 'train'}
# gamma = np.ones(D3)
# beta = np.zeros(D3)
# for t in range(50):
#     X = np.random.randn(N, D1)
#     a = np.maximum(0, X.dot(W1)).dot(W2)  #模拟经过了X*W1 Relu *W2三个步骤
#     batchnorm_forward(a, gamma, beta, bn_param)  #刷新了50次的running_mean和running_variance
# bn_param['mode'] = 'test'
# X = np.random.randn(N, D1)
# a = np.maximum(0, X.dot(W1)).dot(W2)
# a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)
# # Means should be close to zero and stds close to one, but will be
# # noisier than training-time forward passes.
# print ('After batch normalization (test-time):')
# print ('  means: ', a_norm.mean(axis=0))
# print ('  stds: ', a_norm.std(axis=0))


'''--------------backpropagate过程测试----------------------------'''
N, D = 4, 5
x = 5 * np.random.randn(N, D) + 12
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)

bn_param = {'mode': 'train'}
fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
da_num = eval_numerical_gradient_array(fg, gamma, dout)
db_num = eval_numerical_gradient_array(fb, beta, dout)

_, cache = batchnorm_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
print ('dx error: ', rel_error(dx_num, dx))
print ('dgamma error: ', rel_error(da_num, dgamma))
print ('dbeta error: ', rel_error(db_num, dbeta))
