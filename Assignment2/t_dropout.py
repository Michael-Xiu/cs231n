from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
from dropout import *
import numpy as np
import CIFAR_use
import matplotlib.pyplot as plt


#这个包是官方指南里面漏的，注意要补上
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#这个函数是官方指南里面漏的，注意要补上


'''测试dropout前向功能'''
# x = np.random.randn(500, 500) + 10
#
# for p in [0.25, 0.4, 0.7]:
#   out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
#   out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})
#
#   print('Running tests with p = ', p)
#   print('Mean of input: ', x.mean())
#   print('Mean of train-time output: ', out.mean())
#   print('Mean of test-time output: ', out_test.mean())
#   print('Fraction of train-time output set to zero: ', (out == 0).mean())  #训练时设置为0的概率
#   print('Fraction of test-time output set to zero: ', (out_test == 0).mean())  #测试时设置为0的概率
#   print()


'''测试dropout的后向功能'''
x = np.random.randn(10, 10) + 10
dout = np.random.randn(*x.shape)

dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}
out, cache = dropout_forward(x, dropout_param)
dx = dropout_backward(dout, cache)
dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)

# Error should be around e-10 or less
print('dx relative error: ', rel_error(dx, dx_num))

