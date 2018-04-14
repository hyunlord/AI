import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pylab as plt

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

mnist = input_data.read_data_sets("./mnist", one_hot = True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_noise = 128

X = tf.placeholder(tf.float32, [None, 784])
Z = tf.placeholder(tf.float32, [None, n_noise])
keep_prob = tf.placeholder(tf.float32)

filter_1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
filter_2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))

W1 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev = 0.01))
W2 = tf.Variable(tf.random_normal([256, 100], stddev = 0.01))
W3 = tf.Variable(tf.random_normal([100, 1], stddev = 0.01))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([100]))
B3 = tf.Variable(tf.random_normal([1]))

W4 = tf.Variable(tf.random_normal([n_noise, 256], stddev = 0.01))
W5 = tf.Variable(tf.random_normal([256, 400], stddev = 0.01))
W6 = tf.Variable(tf.random_normal([400, 784], stddev = 0.01))

B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([400]))
B6 = tf.Variable(tf.random_normal([784]))

# Discriminator
def Discriminator(input):
    input = tf.reshape(input, [-1, 28, 28, 1])
    
    layer_1 = tf.nn.conv2d(input, filter_1, strides = [1, 1, 1, 1], padding = 'SAME')
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.max_pool(layer_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    layer_2 = tf.nn.conv2d(layer_1, filter_2, strides = [1, 1, 1, 1], padding = 'SAME')
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.max_pool(layer_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')    

    layer_3 = tf.reshape(layer_2, [-1, 7 * 7 * 64])
    layer_3 = tf.matmul(layer_3, W1) + B1
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)

    layer_4 = tf.matmul(layer_3, W2) + B2
    layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.nn.dropout(layer_4, keep_prob)

    model_D = tf.matmul(layer_4, W3) + B3
    model_D = tf.nn.sigmoid(model_D)
    
    return model_D

# Generator
def Generator(noise):
    layer_5 = tf.matmul(noise, W4) + B4
    layer_5 = tf.nn.relu(layer_5)
    #layer_5 = tf.nn.dropout(layer_5, keep_prob)

    layer_6 = tf.matmul(layer_5, W5) + B5
    layer_6 = tf.nn.relu(layer_6)
    #layer_6 = tf.nn.dropout(layer_6, keep_prob)

    model_G = tf.matmul(layer_6, W6) + B6
    model_G = tf.nn.sigmoid(model_G)
    
    return model_G

# Make noise
def makeNoise(batch_size, n_noise):
    return np.random.normal(size = (batch_size, n_noise))

G = Generator(Z)
D_gene = Discriminator(G)
D_real = Discriminator(X)

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

var_D_list = [filter_1, filter_2, W1, W2, W3, B1, B2, B3]
#var_D_list = [W1, W2, W3, B1, B2, B3]
var_G_list = [W4, W5, W6, B4, B5, B6]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list = var_D_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list = var_G_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):    
    for j in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = makeNoise(batch_size, n_noise)
        
        _, loss_val_D = sess.run([train_D, loss_D], feed_dict = {X : batch_xs, Z : noise, keep_prob : 0.7})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict = {Z : noise, keep_prob : 0.7})
        
    print('Epoch : ', '%04d' % epoch,
          'D loss : {:.4}'.format(loss_val_D),
          'G loss : {:.4}'.format(loss_val_G))
        
    if epoch >= 0:
        sample_size = 12
        noise = makeNoise(sample_size, n_noise)
        samples = sess.run(G, feed_dict = {Z : noise, keep_prob : 1})
        
        fig, ax = plt.subplots(1, sample_size, figsize = (sample_size, 1))
        
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))
            
        plt.savefig('samples-conv/{}.png'.format(str(epoch).zfill(3)), bbox_inches = 'tight')
        plt.close(fig)
        
print('최적화 완료')