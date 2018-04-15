import tensorflow as tf
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
import numpy as np
import matplotlib.pylab as plt

(x_train, y_train), (x_test, y_test) = load_data()

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis = 1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis = 1)

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

total_epoch = 25
batch_size = 100
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

filter_1 = tf.Variable(tf.random_normal([5, 5, 3, 36], stddev = 0.01))
filter_2 = tf.Variable(tf.random_normal([5, 5, 36, 96], stddev = 0.01))

W1 = tf.Variable(tf.random_normal([8 * 8 * 96, 900], stddev = 0.01))
W2 = tf.Variable(tf.random_normal([900, 400], stddev = 0.01))
W3 = tf.Variable(tf.random_normal([400, 100], stddev = 0.01))
W4 = tf.Variable(tf.random_normal([100, 10], stddev = 0.01))

B1 = tf.Variable(tf.random_normal([900], stddev = 0.01))
B2 = tf.Variable(tf.random_normal([400], stddev = 0.01))
B3 = tf.Variable(tf.random_normal([100], stddev = 0.01))
B4 = tf.Variable(tf.random_normal([10], stddev = 0.01))

conv_1 = tf.nn.conv2d(X, filter_1, strides = [1, 1, 1, 1], padding = 'SAME')
conv_1 = tf.nn.relu(conv_1)
conv_1 = tf.nn.dropout(conv_1, keep_prob)
conv_1 = tf.nn.max_pool(conv_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                         padding = 'SAME')

conv_2 = tf.nn.conv2d(conv_1, filter_2, strides = [1, 1, 1, 1], padding = 'SAME')
conv_2 = tf.nn.relu(conv_2)
conv_2 = tf.nn.dropout(conv_2, keep_prob)
conv_2 = tf.nn.max_pool(conv_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                         padding = 'SAME')
conv_2 = tf.reshape(conv_2, [-1, 8 * 8 * 96])

layer_1 = tf.matmul(conv_2, W1) + B1
layer_1 = tf.nn.relu(layer_1)
layer_1 = tf.nn.dropout(layer_1, keep_prob)

layer_2 = tf.matmul(layer_1, W2) + B2
layer_2 = tf.nn.relu(layer_2)
layer_2 = tf.nn.dropout(layer_2, keep_prob)

layer_3 = tf.matmul(layer_2, W3) + B3
layer_3 = tf.nn.relu(layer_3)
layer_3 = tf.nn.dropout(layer_3, keep_prob)

model = tf.matmul(layer_3, W4) + B4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits = model, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

total_batch = x_train.shape[0] // batch_size

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(total_epoch):
        total_cost = 0
    
        for i in range(total_batch):
            batch = next_batch(batch_size, x_train, y_train_one_hot.eval())
        
            _, loss = sess.run([optimizer, cost], feed_dict = {
                    X : batch[0], Y : batch[1], keep_prob : 0.5})
            total_cost += loss
            
        print('Epoch : ', '%04d' % (epoch + 1),
              'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))
        train_accuracy = accuracy.eval(feed_dict={X : batch[0], Y: batch[1], keep_prob: 1.0})
        loss_print = cost.eval(feed_dict={X : batch[0], Y : batch[1], keep_prob: 1.0})

        print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (epoch + 1, 
              train_accuracy, loss_print))
        
    test_batch = next_batch(10000, x_test, y_test_one_hot.eval())
    print("테스트 데이터 정확도: %f" % accuracy.eval(feed_dict={X: test_batch[0],
                                                       Y: test_batch[1], keep_prob: 1.0}))