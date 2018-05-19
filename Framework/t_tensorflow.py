import numpy as np
np.random.seed(0)
import tensorflow as tf
import time #计时


t0 = time.clock()

N,D = 3000, 4000

with tf.device('/gpu:0'):
    x=tf.placeholder(tf.float32)
    y=tf.placeholder(tf.float32)
    z=tf.placeholder(tf.float32)

    a=x*y
    b=a+z
    c=tf.reduce_sum(b)

grad_x,grad_y,grad_z=tf.gradients(c,[x,y,z])

with tf.Session() as sess:
    values={
        x:np.random.randn(N,D),
        y: np.random.randn(N, D),
        z: np.random.randn(N, D),
    }
    out=sess.run([c,grad_x,grad_y,grad_z],feed_dict=values)
    c_val,grad_x_val,grad_y_val,grad_z_val=out

print(grad_x)
print(grad_y)
print(grad_z)

print (time.clock() - t0)


# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))
