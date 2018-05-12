from cs231n.solver import *
from FNN import *
import FNN
import matplotlib.pyplot as plt

'''
关于solver:
model是自己定义的模型，data是数据，update_relu是选择的优化器,optim_config是所需要的参数,
lr_decay是学习率再每个epoch之后的倍数,num_epochs轮数，指我们的数据要训练多少次，
batch_size是minibatch大小，print——every指每几次迭代输出损失函数值
'''

#测试两层神经网络
#
# model=TwoLayerNet(reg=1e-1)
# [X_train,y_train,X_val,y_val,X_test,y_test]=FNN.get_cifar_data(num_training=49000,num_validation=1000,num_test=1000)
# data ={'X_train':X_train,
#        'y_train':y_train,
#        'X_val':X_val,
#        'y_val':y_val,
#        'X_test':X_test,
#        'y_test':y_test}
#
# solver=Solver(model,data,update_rule='sgd',optim_config={'learning_rate':1e-3},lr_decay=0.8,num_epochs=10,batch_size=100,print_every=100)
#
# solver.train()
# scores=model.loss(data['X_test'])
# y_pred=np.argmax(scores,axis=1)
# acc=np.mean(y_pred==data['y_test'])
# print('test acc: %f'%acc)
#
#
# plt.title('Training loss')
# plt.xlabel('Iteration')
# plt.plot(solver.loss_history)
# plt.show()

''' 两层神经网络训练结果
F:\Anaconda\data\python.exe "C:/Users/Michael/Desktop/Project/CNN/Sranford CNN/homework/assignment2/t2_solver.py"
(Iteration 1 / 4900) loss: 2.318536
(Epoch 0 / 10) train acc: 0.144000; val_acc: 0.152000
(Iteration 101 / 4900) loss: 1.853529
(Iteration 201 / 4900) loss: 1.584651
(Iteration 301 / 4900) loss: 1.632486
(Iteration 401 / 4900) loss: 1.556711
(Epoch 1 / 10) train acc: 0.451000; val_acc: 0.418000
(Iteration 501 / 4900) loss: 1.828123
(Iteration 601 / 4900) loss: 1.534536
(Iteration 701 / 4900) loss: 1.516372
(Iteration 801 / 4900) loss: 1.283637
(Iteration 901 / 4900) loss: 1.419327
(Epoch 2 / 10) train acc: 0.479000; val_acc: 0.469000
(Iteration 1001 / 4900) loss: 1.559862
(Iteration 1101 / 4900) loss: 1.395021
(Iteration 1201 / 4900) loss: 1.563385
(Iteration 1301 / 4900) loss: 1.310555
(Iteration 1401 / 4900) loss: 1.459277
(Epoch 3 / 10) train acc: 0.514000; val_acc: 0.497000
(Iteration 1501 / 4900) loss: 1.302668
(Iteration 1601 / 4900) loss: 1.352934
(Iteration 1701 / 4900) loss: 1.251133
(Iteration 1801 / 4900) loss: 1.397829
(Iteration 1901 / 4900) loss: 1.334595
(Epoch 4 / 10) train acc: 0.544000; val_acc: 0.502000
(Iteration 2001 / 4900) loss: 1.239752
(Iteration 2101 / 4900) loss: 1.456161
(Iteration 2201 / 4900) loss: 1.332228
(Iteration 2301 / 4900) loss: 1.485141
(Iteration 2401 / 4900) loss: 1.258366
(Epoch 5 / 10) train acc: 0.554000; val_acc: 0.498000
(Iteration 2501 / 4900) loss: 1.361256
(Iteration 2601 / 4900) loss: 1.198355
(Iteration 2701 / 4900) loss: 1.434318
(Iteration 2801 / 4900) loss: 1.291840
(Iteration 2901 / 4900) loss: 1.101839
(Epoch 6 / 10) train acc: 0.572000; val_acc: 0.510000
(Iteration 3001 / 4900) loss: 1.286684
(Iteration 3101 / 4900) loss: 1.283414
(Iteration 3201 / 4900) loss: 1.095553
(Iteration 3301 / 4900) loss: 1.312107
(Iteration 3401 / 4900) loss: 1.121176
(Epoch 7 / 10) train acc: 0.594000; val_acc: 0.520000
(Iteration 3501 / 4900) loss: 1.187854
(Iteration 3601 / 4900) loss: 1.166775
(Iteration 3701 / 4900) loss: 1.085371
(Iteration 3801 / 4900) loss: 1.065152
(Iteration 3901 / 4900) loss: 1.380333
(Epoch 8 / 10) train acc: 0.592000; val_acc: 0.518000
(Iteration 4001 / 4900) loss: 1.223247
(Iteration 4101 / 4900) loss: 1.391141
(Iteration 4201 / 4900) loss: 1.205393
(Iteration 4301 / 4900) loss: 1.141160
(Iteration 4401 / 4900) loss: 1.135516
(Epoch 9 / 10) train acc: 0.574000; val_acc: 0.527000
(Iteration 4501 / 4900) loss: 1.055002
(Iteration 4601 / 4900) loss: 0.981879
(Iteration 4701 / 4900) loss: 1.068581
(Iteration 4801 / 4900) loss: 1.127199
(Epoch 10 / 10) train acc: 0.596000; val_acc: 0.524000
test acc: 0.522000
'''


#测试三层神经网络
#
model=ThreeLayerNet(weight_scale=1e-2,reg=1e-1)
[X_train,y_train,X_val,y_val,X_test,y_test]=FNN.get_cifar_data(num_training=49000,num_validation=1000,num_test=1000)
data ={'X_train':X_train,
       'y_train':y_train,
       'X_val':X_val,
       'y_val':y_val,
       'X_test':X_test,
       'y_test':y_test}

solver=Solver(model,data,update_rule='sgd',optim_config={'learning_rate':8e-3},lr_decay=0.7,num_epochs=20,batch_size=100,print_every=100)

solver.train()
scores=model.loss(data['X_test'])
y_pred=np.argmax(scores,axis=1)
acc=np.mean(y_pred==data['y_test'])
print('test acc: %f'%acc)

plt.title('Training loss')
plt.xlabel('Iteration')
plt.plot(solver.loss_history)
plt.show()
''' 三层神经网络训练结果
(Iteration 1 / 9800) loss: 3.872213
(Epoch 0 / 20) train acc: 0.130000; val_acc: 0.128000
(Iteration 101 / 9800) loss: 3.229630
(Iteration 201 / 9800) loss: 2.773768
(Iteration 301 / 9800) loss: 2.575975
(Iteration 401 / 9800) loss: 2.674686
(Epoch 1 / 20) train acc: 0.442000; val_acc: 0.444000
(Iteration 501 / 9800) loss: 2.303785
(Iteration 601 / 9800) loss: 2.428197
(Iteration 701 / 9800) loss: 2.163971
(Iteration 801 / 9800) loss: 2.110539
(Iteration 901 / 9800) loss: 2.006152
(Epoch 2 / 20) train acc: 0.500000; val_acc: 0.479000
(Iteration 1001 / 9800) loss: 1.922030
(Iteration 1101 / 9800) loss: 1.970117
(Iteration 1201 / 9800) loss: 1.897506
(Iteration 1301 / 9800) loss: 1.928037
(Iteration 1401 / 9800) loss: 1.652994
(Epoch 3 / 20) train acc: 0.513000; val_acc: 0.493000
(Iteration 1501 / 9800) loss: 1.738816
(Iteration 1601 / 9800) loss: 1.738450
(Iteration 1701 / 9800) loss: 1.866370
(Iteration 1801 / 9800) loss: 1.666660
(Iteration 1901 / 9800) loss: 1.627670
(Epoch 4 / 20) train acc: 0.526000; val_acc: 0.497000
(Iteration 2001 / 9800) loss: 1.524532
(Iteration 2101 / 9800) loss: 1.702236
(Iteration 2201 / 9800) loss: 1.574647
(Iteration 2301 / 9800) loss: 1.355378
(Iteration 2401 / 9800) loss: 1.583107
(Epoch 5 / 20) train acc: 0.548000; val_acc: 0.513000
(Iteration 2501 / 9800) loss: 1.474042
(Iteration 2601 / 9800) loss: 1.573469
(Iteration 2701 / 9800) loss: 1.465497
(Iteration 2801 / 9800) loss: 1.507603
(Iteration 2901 / 9800) loss: 1.558829
(Epoch 6 / 20) train acc: 0.530000; val_acc: 0.506000
(Iteration 3001 / 9800) loss: 1.439790
(Iteration 3101 / 9800) loss: 1.502802
(Iteration 3201 / 9800) loss: 1.674163
(Iteration 3301 / 9800) loss: 1.345039
(Iteration 3401 / 9800) loss: 1.304950
(Epoch 7 / 20) train acc: 0.556000; val_acc: 0.517000
(Iteration 3501 / 9800) loss: 1.248820
(Iteration 3601 / 9800) loss: 1.514967
(Iteration 3701 / 9800) loss: 1.437206
(Iteration 3801 / 9800) loss: 1.386580
(Iteration 3901 / 9800) loss: 1.398283
(Epoch 8 / 20) train acc: 0.590000; val_acc: 0.524000
(Iteration 4001 / 9800) loss: 1.251978
(Iteration 4101 / 9800) loss: 1.435634
(Iteration 4201 / 9800) loss: 1.300541
(Iteration 4301 / 9800) loss: 1.437080
(Iteration 4401 / 9800) loss: 1.517465
(Epoch 9 / 20) train acc: 0.588000; val_acc: 0.529000
(Iteration 4501 / 9800) loss: 1.504726
(Iteration 4601 / 9800) loss: 1.332455
(Iteration 4701 / 9800) loss: 1.583334
(Iteration 4801 / 9800) loss: 1.257256
(Epoch 10 / 20) train acc: 0.578000; val_acc: 0.535000
(Iteration 4901 / 9800) loss: 1.294506
(Iteration 5001 / 9800) loss: 1.319056
(Iteration 5101 / 9800) loss: 1.376785
(Iteration 5201 / 9800) loss: 1.422026
(Iteration 5301 / 9800) loss: 1.435851
(Epoch 11 / 20) train acc: 0.622000; val_acc: 0.540000
(Iteration 5401 / 9800) loss: 1.400996
(Iteration 5501 / 9800) loss: 1.490957
(Iteration 5601 / 9800) loss: 1.176893
(Iteration 5701 / 9800) loss: 1.197258
(Iteration 5801 / 9800) loss: 1.400555
(Epoch 12 / 20) train acc: 0.622000; val_acc: 0.548000
(Iteration 5901 / 9800) loss: 1.158974
(Iteration 6001 / 9800) loss: 1.278672
(Iteration 6101 / 9800) loss: 1.234902
(Iteration 6201 / 9800) loss: 1.306296
(Iteration 6301 / 9800) loss: 1.241346
(Epoch 13 / 20) train acc: 0.613000; val_acc: 0.542000
(Iteration 6401 / 9800) loss: 1.331448
(Iteration 6501 / 9800) loss: 1.389011
(Iteration 6601 / 9800) loss: 1.475358
(Iteration 6701 / 9800) loss: 1.282285
(Iteration 6801 / 9800) loss: 1.181638
(Epoch 14 / 20) train acc: 0.590000; val_acc: 0.550000
(Iteration 6901 / 9800) loss: 1.370799
(Iteration 7001 / 9800) loss: 1.335567
(Iteration 7101 / 9800) loss: 1.215389
(Iteration 7201 / 9800) loss: 1.325665
(Iteration 7301 / 9800) loss: 1.103510
(Epoch 15 / 20) train acc: 0.648000; val_acc: 0.544000
(Iteration 7401 / 9800) loss: 1.370379
(Iteration 7501 / 9800) loss: 1.329029
(Iteration 7601 / 9800) loss: 1.398211
(Iteration 7701 / 9800) loss: 1.304058
(Iteration 7801 / 9800) loss: 1.251347
(Epoch 16 / 20) train acc: 0.624000; val_acc: 0.549000
(Iteration 7901 / 9800) loss: 1.270315
(Iteration 8001 / 9800) loss: 1.241240
(Iteration 8101 / 9800) loss: 1.389193
(Iteration 8201 / 9800) loss: 1.313566
(Iteration 8301 / 9800) loss: 1.311301
(Epoch 17 / 20) train acc: 0.629000; val_acc: 0.554000
(Iteration 8401 / 9800) loss: 1.310647
(Iteration 8501 / 9800) loss: 1.240528
(Iteration 8601 / 9800) loss: 1.162984
(Iteration 8701 / 9800) loss: 1.412341
(Iteration 8801 / 9800) loss: 1.425771
(Epoch 18 / 20) train acc: 0.631000; val_acc: 0.560000
(Iteration 8901 / 9800) loss: 1.216988
(Iteration 9001 / 9800) loss: 1.262856
(Iteration 9101 / 9800) loss: 1.297891
(Iteration 9201 / 9800) loss: 1.170397
(Iteration 9301 / 9800) loss: 1.326448
(Epoch 19 / 20) train acc: 0.638000; val_acc: 0.548000
(Iteration 9401 / 9800) loss: 1.340005
(Iteration 9501 / 9800) loss: 1.490923
(Iteration 9601 / 9800) loss: 1.114468
(Iteration 9701 / 9800) loss: 1.187624
(Epoch 20 / 20) train acc: 0.672000; val_acc: 0.551000
test acc: 0.550000
'''

