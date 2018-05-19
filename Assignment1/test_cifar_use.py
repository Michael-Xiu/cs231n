import CIFAR_use
x_train,y_train,x_test,y_test=CIFAR_use.load_cifar10('C:/Users/Michael/Desktop/Project/CNN/Sranford CNN/homework/CIFAR-10')
print('training data shape:',x_train.shape)
print('training labels shape:',y_train.shape)
print('test data shape:',x_test.shape)
print('test labels shape:',y_test.shape)