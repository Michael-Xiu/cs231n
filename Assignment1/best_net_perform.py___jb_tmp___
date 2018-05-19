import TwoLayerNN
import numpy as np

#交叉验证最好的结果为 中间层100变量 learning_rate=1e-3 正则项为0.75


input_size=32*32*3  #3072
hidden_size=[100]   #选项1 中间层的参量数
num_classes=10
results={}
best_val_acc=0
best_net=None

learning_rates=np.array([1])*1e-3  #选项2 下降的速度
regularization_strengths=[0.75]  #选项3 正则项大小

print('running')

for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            net=TwoLayerNN.Neural_Network(input_size,hs,num_classes)

            [X_train, y_train, X_val, y_val, X_test, y_test] = net.get_cifar_data(num_training=49000,
                                                                                  num_validation=1000, num_test=10000)
            stats=net.train(X_train,y_train,X_val,y_val,num_iters=1500,batch_size=200,learning_rate=lr,
                learning_rate_decay=0.95,reg=reg,verbose=False)
            val_acc = np.mean((net.predict(X_val) == y_val))  # 计算模型交叉验证的准确度
            if val_acc>best_val_acc:
                best_val_acc=val_acc
                best_net=net
            results[(hs,lr,reg)]=val_acc   #对于每一个组合，写出它的预测准确度


print('finished')

for hs,lr,reg in sorted(results):
    val_acc=results[(hs,lr,reg)]
    print('hs %d lr %e reg %e val accuracy: %f'%(hs,lr,reg,val_acc))

print('best validation accuracy achieved during cross_validation %f' % best_val_acc)

TwoLayerNN.show_net_weights(best_net)  #画出分类特征

test_acc=np.mean((net.predict(X_test)==y_test))  # 计算模型在10000个测试图像的准确度
print('testing batch accuracy:',test_acc)





