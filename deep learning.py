import pandas as pd
import preprocess
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer


data = pd.read_csv("myFP_217_D2.csv", header=None)

D2 = preprocess.get_data(data)

X = preprocess.get_X(D2)
# y = pd.DataFrame(preprocess.get_target(D2))
value = preprocess.get_target(D2)
value = MultiLabelBinarizer().fit_transform(value)
y = pd.DataFrame(value)

X_train, X_test = X[:int((0.8*len(X)))], X[int((0.8*len(X))):]
y_train, y_test = y[:int((0.8*len(X)))], y[int((0.8*len(X))):]


def basicnn():

    #  def nn layer
    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function == None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output



    # set up placeholder
    xs = tf.placeholder(tf.float32, [None, 2048], name="features")
    ys = tf.placeholder(tf.float32, [None, 7], name="targets")

    # set up structure
    l1 = add_layer(xs, 2048, 1024, activation_function=tf.nn.sigmoid)  # relu --> cross
    # l2 = add_layer(l1, 1024, 512, activation_function=tf.nn.sigmoid)
    # l3 = add_layer(l2, 512, 256, activation_function=tf.nn.sigmoid)
    # l4 = add_layer(l3, 256, 128, activation_function=tf.nn.sigmoid)
    # l5 = add_layer(l4, 128, 64, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l1, 1024, 7, activation_function=tf.nn.sigmoid)

    # loss function
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)

    for i in xrange(0, 10000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
        if i % 100 == 0:
            print 'process {0}'.format(i)
            print '--Accuracy:--', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train})
            print 'Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test})
            print 'cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train})
            saver.save(sess, 'basic2')

def basicnn2():

    #  def nn layer
    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function == None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output



    # set up placeholder
    xs = tf.placeholder(tf.float32, [None, 2048], name="features")
    ys = tf.placeholder(tf.float32, [None, 7], name="targets")

    # set up structure
    l1 = add_layer(xs, 2048, 1024, activation_function=tf.nn.sigmoid)  # relu --> cross
    l2 = add_layer(l1, 1024, 512, activation_function=tf.nn.sigmoid)
    # l3 = add_layer(l2, 512, 256, activation_function=tf.nn.sigmoid)
    # l4 = add_layer(l3, 256, 128, activation_function=tf.nn.sigmoid)
    # l5 = add_layer(l4, 128, 64, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l2, 512, 7, activation_function=tf.nn.sigmoid)

    # loss function
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    for i in xrange(0, 10000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
        if i % 100 == 0:
            print 'process {0}'.format(i)
            print '--Accuracy:--', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train})
            print 'Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test})
            print 'cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train})
            saver.save(sess, 'basic2')

def basicnn3():

    #  def nn layer
    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function == None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output



    # set up placeholder
    xs = tf.placeholder(tf.float32, [None, 2048], name="features")
    ys = tf.placeholder(tf.float32, [None, 7], name="targets")

    # set up structure
    l1 = add_layer(xs, 2048, 1024, activation_function=tf.nn.sigmoid)  # relu --> cross
    l2 = add_layer(l1, 1024, 512, activation_function=tf.nn.sigmoid)
    l3 = add_layer(l2, 512, 256, activation_function=tf.nn.sigmoid)
    # l4 = add_layer(l3, 256, 128, activation_function=tf.nn.sigmoid)
    # l5 = add_layer(l4, 128, 64, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l3, 1024, 7, activation_function=tf.nn.sigmoid)

    # loss function
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)

    for i in xrange(0, 10000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
        if i % 100 == 0:
            print 'process {0}'.format(i)
            print '--Accuracy:--', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train})
            print 'Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test})
            print 'cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train})
            saver.save(sess, 'basic2')

def basicnn4():

    #  def nn layer
    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function == None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output



    # set up placeholder
    xs = tf.placeholder(tf.float32, [None, 2048], name="features")
    ys = tf.placeholder(tf.float32, [None, 7], name="targets")

    # set up structure
    l1 = add_layer(xs, 2048, 1024, activation_function=tf.nn.sigmoid)  # relu --> cross
    l2 = add_layer(l1, 1024, 512, activation_function=tf.nn.sigmoid)
    l3 = add_layer(l2, 512, 256, activation_function=tf.nn.sigmoid)
    l4 = add_layer(l3, 256, 128, activation_function=tf.nn.sigmoid)
    # l5 = add_layer(l4, 128, 64, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l1, 128, 7, activation_function=tf.nn.sigmoid)

    # loss function
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)

    for i in xrange(0, 10000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
        if i % 100 == 0:
            print 'process {0}'.format(i)
            print '--Accuracy:--', sess.run(accuracy, feed_dict={xs: X_train, ys: y_train})
            print 'Accuracy:', sess.run(accuracy, feed_dict={xs: X_test, ys: y_test})
            print 'cost:', sess.run(cost, feed_dict={xs: X_train, ys: y_train})
            saver.save(sess, 'basic2')

if __name__ == "__main__":
    # basicnn()
    basicnn2()
    # basicnn3()
    # basicnn4()


