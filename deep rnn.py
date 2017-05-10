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

def rnn1():
    n_nodes_hl1 = 1024
    n_nodes_hl2 = 512
    n_nodes_hl3 = 256

    n_classes = 7
    batch_size = 100

    x = tf.placeholder(tf.float32, [None, 2048], name="features")
    y = tf.placeholder(tf.float32, [None, 7], name="targets")


    def neural_network_model(data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([2048, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes])), }

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.tanh(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.tanh(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.tanh(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

        return output


    def train_neural_network(x):
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        hm_epochs = 100
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in xrange(0, 100):
                    epoch_x, epoch_y = X_train,y_train
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))
                accuracy1 = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Train Accuracy:', accuracy1.eval({x: X_train, y: y_train}))

    train_neural_network(x)


def rnn2():
    n_nodes_hl1 = 512
    n_nodes_hl2 = 256
    n_nodes_hl3 = 128

    n_classes = 7
    batch_size = 100

    x = tf.placeholder(tf.float32, [None, 2048], name="features")
    y = tf.placeholder(tf.float32, [None, 7], name="targets")

    def neural_network_model(data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([2048, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes])), }

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.tanh(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.tanh(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.tanh(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

        return output

    def train_neural_network(x):
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        hm_epochs = 100
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in xrange(0, 100):
                    epoch_x, epoch_y = X_train, y_train
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))
                accuracy1 = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Train Accuracy:', accuracy1.eval({x: X_train, y: y_train}))

    train_neural_network(x)


def rnn3():
    n_nodes_hl1 = 512
    n_nodes_hl2 = 128
    n_nodes_hl3 = 64

    n_classes = 7
    batch_size = 100

    x = tf.placeholder(tf.float32, [None, 2048], name="features")
    y = tf.placeholder(tf.float32, [None, 7], name="targets")

    def neural_network_model(data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([2048, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes])), }

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.tanh(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.tanh(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.tanh(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

        return output

    def train_neural_network(x):
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        hm_epochs = 100
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in xrange(0, 100):
                    epoch_x, epoch_y = X_train, y_train
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))
                accuracy1 = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Train Accuracy:', accuracy1.eval({x: X_train, y: y_train}))



    train_neural_network(x)


def rnn4():
    n_nodes_hl1 = 256
    n_nodes_hl2 = 128
    n_nodes_hl3 = 32

    n_classes = 7
    batch_size = 100

    x = tf.placeholder(tf.float32, [None, 2048], name="features")
    y = tf.placeholder(tf.float32, [None, 7], name="targets")

    def neural_network_model(data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([2048, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes])), }

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.tanh(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.tanh(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.tanh(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

        return output

    def train_neural_network(x):
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        hm_epochs = 100
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in xrange(0, 100):
                    epoch_x, epoch_y = X_train, y_train
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))
                accuracy1 = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Train Accuracy:', accuracy1.eval({x: X_train, y: y_train}))

    train_neural_network(x)


if __name__ == "__main__":
    # rnn1()
    rnn2()
    # rnn3()
    # rnn4()
