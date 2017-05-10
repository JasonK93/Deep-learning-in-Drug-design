import pandas as pd
import preprocess
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle


data = pd.read_csv("myFP_217_D2.csv", header=None)

D2 = preprocess.get_data(data)

X = preprocess.get_X(D2)
# y = pd.DataFrame(preprocess.get_target(D2))
value = preprocess.get_target(D2)
value = MultiLabelBinarizer().fit_transform(value)
y = pd.DataFrame(value)

X, y = shuffle(X, y, random_state=0)

X_train, X_test = X[:int((0.8*len(X)))], X[int((0.8*len(X))):]
y_train, y_test = y[:int((0.8*len(X)))], y[int((0.8*len(X))):]


def cnn1():
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv1d(x, W):
        return tf.nn.conv1d(x, W, stride=2, padding='SAME')

    xs = tf.placeholder(tf.float32, [None, 2048], name="features")
    ys = tf.placeholder(tf.float32, [None, 7], name="targets")

    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 2048, 1])
    ## conv1 layer ##
    # [filter_width, in_channels, out_channels]
    W_conv1 = weight_variable([8, 1, 4])
    b_conv1 = bias_variable([4])
    h_conv1 = tf.nn.tanh(conv1d(x_image, W_conv1) + b_conv1)  # 1024x4

    ## conv2 layer ##
    # [filter_width, in_channels, out_channels]
    W_conv2 = weight_variable([8, 4, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.tanh(conv1d(h_conv1, W_conv2) + b_conv2)  # 512x16

    ## func1 layer ##
    W_fc1 = weight_variable([512 * 16, 256])
    b_fc1 = bias_variable([256])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 512 * 16])
    h_cf1 = tf.nn.sigmoid(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_cf1, keep_prob)

    ## func2 layer ##
    W_fc2 = weight_variable([256, 7])
    b_fc2 = bias_variable([7])
    # prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc1)


    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the error between prediction and real data
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in xrange(0, 10000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.8})
        if i % 100 == 0:

            print 'process {0}'.format(i)
            print '--Accuracy:--', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            print 'Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            print 'cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})

def cnn2():
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv1d(x, W):
        return tf.nn.conv1d(x, W, stride=2, padding='SAME')

    xs = tf.placeholder(tf.float32, [None, 2048], name="features")
    ys = tf.placeholder(tf.float32, [None, 7], name="targets")

    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 2048, 1])
    ## conv1 layer ##
    # [filter_width, in_channels, out_channels]
    W_conv1 = weight_variable([16, 1, 2])
    b_conv1 = bias_variable([2])
    h_conv1 = tf.nn.tanh(conv1d(x_image, W_conv1) + b_conv1)  # 1024x2

    ## conv2 layer ##
    # [filter_width, in_channels, out_channels]
    W_conv2 = weight_variable([16, 2, 4])
    b_conv2 = bias_variable([4])
    h_conv2 = tf.nn.tanh(conv1d(h_conv1, W_conv2) + b_conv2)  # 512x4

    ## func1 layer ##
    W_fc1 = weight_variable([512 * 4, 128])
    b_fc1 = bias_variable([128])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 512 * 4])
    h_cf1 = tf.nn.sigmoid(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_cf1, keep_prob)

    ## func2 layer ##
    W_fc2 = weight_variable([128, 7])
    b_fc2 = bias_variable([7])
    # prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc1)


    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the error between prediction and real data
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in xrange(0, 10000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.8})
        if i % 100 == 0:
            print 'process {0}'.format(i)
            print '--Accuracy:--', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            print 'Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            print 'cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})

def cnn3():
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv1d(x, W):
        return tf.nn.conv1d(x, W, stride=4, padding='SAME')

    xs = tf.placeholder(tf.float32, [None, 2048], name="features")
    ys = tf.placeholder(tf.float32, [None, 7], name="targets")

    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 2048, 1])
    ## conv1 layer ##
    # [filter_width, in_channels, out_channels]
    W_conv1 = weight_variable([16, 1, 4])
    b_conv1 = bias_variable([4])
    h_conv1 = tf.nn.tanh(conv1d(x_image, W_conv1) + b_conv1)  # 512x4

    ## conv2 layer ##
    # [filter_width, in_channels, out_channels]
    W_conv2 = weight_variable([16, 4, 8])
    b_conv2 = bias_variable([8])
    h_conv2 = tf.nn.tanh(conv1d(h_conv1, W_conv2) + b_conv2)  # 128x8

    ## func1 layer ##
    W_fc1 = weight_variable([128 * 8, 128])
    b_fc1 = bias_variable([128])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 128 * 8])
    h_cf1 = tf.nn.sigmoid(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_cf1, keep_prob)

    ## func2 layer ##
    W_fc2 = weight_variable([128, 7])
    b_fc2 = bias_variable([7])
    # prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc1)


    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the error between prediction and real data
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in xrange(0, 10000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.8})
        if i % 100 == 0:
            print 'process {0}'.format(i)
            print '--Accuracy:--', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            print 'Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            print 'cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})

def cnn4():
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv1d(x, W):
        return tf.nn.conv1d(x, W, stride=8, padding='SAME')

    xs = tf.placeholder(tf.float32, [None, 2048], name="features")
    ys = tf.placeholder(tf.float32, [None, 7], name="targets")

    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 2048, 1])
    ## conv1 layer ##
    # [filter_width, in_channels, out_channels]
    W_conv1 = weight_variable([32, 1, 8])
    b_conv1 = bias_variable([8])
    h_conv1 = tf.nn.tanh(conv1d(x_image, W_conv1) + b_conv1)  # 256x8

    ## conv2 layer ##
    # [filter_width, in_channels, out_channels]
    W_conv2 = weight_variable([32, 8, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.tanh(conv1d(h_conv1, W_conv2) + b_conv2)  # 32x16

    ## func1 layer ##
    W_fc1 = weight_variable([32 * 16, 128])
    b_fc1 = bias_variable([128])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 32 * 16])
    h_cf1 = tf.nn.sigmoid(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_cf1, keep_prob)

    ## func2 layer ##
    W_fc2 = weight_variable([128, 7])
    b_fc2 = bias_variable([7])
    # prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc1)


    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the error between prediction and real data
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in xrange(0, 10000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.8})
        if i % 100 == 0:
            print 'process {0}'.format(i)
            print '--Accuracy:--', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            print 'Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            print 'cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})

if __name__ == "__main__":
    cnn1()
    # cnn2()
    # cnn3()
    # cnn4()