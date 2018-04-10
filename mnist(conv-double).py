import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

mnist = input_data.read_data_sets('./mnist', one_hot = True)

input = tf.placeholder(tf.float32, [None, 28, 28, 1])
output = tf.placeholder(tf.float32, [None , 10])
keep_prob = tf.placeholder(tf.float32)

# 1 - [None, 14, 14, 32]
filter_1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
layer_1 = tf.nn.conv2d(input, filter_1, strides = [1, 1, 1, 1], padding = 'SAME')
layer_1 = tf.nn.relu(layer_1)
layer_1 = tf.nn.max_pool(layer_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                         padding = 'SAME')

# 1 - [None, 7, 7, 64]
filter_2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
layer_2 = tf.nn.conv2d(layer_1, filter_2, strides = [1, 1, 1, 1], padding = 'SAME')
layer_2 = tf.nn.relu(layer_2)
layer_2 = tf.nn.max_pool(layer_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                         padding = 'SAME')

W1 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev = 0.01))
W2 = tf.Variable(tf.random_normal([256, 100], stddev = 0.01))
W3 = tf.Variable(tf.random_normal([100, 10], stddev = 0.01))

layer_3 = tf.reshape(layer_2, [-1, 7 * 7 * 64])
layer_3 = tf.matmul(layer_3, W1)
layer_3 = tf.nn.relu(layer_3)
layer_3 = tf.nn.dropout(layer_3, keep_prob)

layer_4 = tf.matmul(layer_3, W2)
layer_4 = tf.nn.relu(layer_4)
layer_4 = tf.nn.dropout(layer_4, keep_prob)

# 2 - [None, 14, 14, 32]
filter_5 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
layer_5 = tf.nn.conv2d(input, filter_5, strides = [1, 1, 1, 1], padding = 'SAME')
layer_5 = tf.nn.relu(layer_5)
layer_5 = tf.nn.max_pool(layer_5, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                         padding = 'SAME')

# [None, 7, 7, 64]
filter_6 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
layer_6 = tf.nn.conv2d(layer_5, filter_6, strides = [1, 1, 1, 1], padding = 'SAME')
layer_6 = tf.nn.relu(layer_6)
layer_6 = tf.nn.max_pool(layer_6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                         padding = 'SAME')

W4 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev = 0.01))
W5 = tf.Variable(tf.random_normal([256, 100], stddev = 0.01))
W6 = tf.Variable(tf.random_normal([100, 10], stddev = 0.01))

layer_7 = tf.reshape(layer_6, [-1, 7 * 7 * 64])
layer_7 = tf.matmul(layer_7, W4)
layer_7 = tf.nn.relu(layer_7)
layer_7 = tf.nn.dropout(layer_7, keep_prob)

layer_8 = tf.matmul(layer_7, W5)
layer_8 = tf.nn.relu(layer_8)
layer_8 = tf.nn.dropout(layer_8, keep_prob)

model_1 = tf.matmul(layer_4, W3)
model_2 = tf.matmul(layer_8, W6)

model = tf.div(tf.add(model_1, model_2), 2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits = model, labels = output))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)        
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)        
        
        _, cost_val = sess.run([optimizer, cost],
            feed_dict = {input : batch_xs, output : batch_ys, keep_prob : 0.8})
        
        total_cost += cost_val
    
    print('Epoch : ', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(output, 1))

# tf.cast를 통해 is_correct를 0과 1로 변환한다.
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('정확도 : ', sess.run(accuracy,
            feed_dict = {input : mnist.test.images.reshape(-1, 28, 28, 1),
                         output : mnist.test.labels,
                         keep_prob : 1}))