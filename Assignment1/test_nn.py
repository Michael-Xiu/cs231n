import TwoLayerNN
import numpy as np


input_size=32*32*3  #3072
hidden_size=50
num_classes=10

net=TwoLayerNN.Neural_Network(input_size,hidden_size,num_classes)
[X_train,y_train,X_val,y_val,X_test,y_test]=net.get_cifar_data(num_training=49000,num_validation=1000,num_test=1000)
stats=net.train(X_train,y_train,X_val,y_val,num_iters=1000,batch_size=200,learning_rate=1e-4,
                learning_rate_decay=0.95,reg=0.5,verbose=True)

val_acc=np.mean((net.predict(X_val)==y_val))  # 计算模型交叉验证的准确度
print('valiadation accuracy:',val_acc)

#net.visualize_loss(stats) #可视化loss和accura

TwoLayerNN.show_net_weights(net)