from numpy.core.fromnumeric import shape
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from numpy import array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.compat.v1.disable_eager_execution()
xin = tf.placeholder('float', [None, 9])
yin = tf.placeholder('float')

df = pd.read_csv('fertilizer.csv')
x = df.drop('class', axis=1)
y = df['class']

values = array(y)

# integer encode

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# binary encode

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=False)

# neural network parameters

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 4
batch_size = 100
data_index = 0


# generate batch

def generate_batch(batch_size):
    global data_index
    # the same shapes as train data
    batch = np.ndarray(shape=(batch_size, 9), dtype=np.float32)
    labels = np.ndarray(shape=(batch_size, 4), dtype=np.float32)
    for i in range(batch_size):
        batch[i] = np.array(x_train)[data_index]
        labels[i] = y_train[data_index]
        data_index = (data_index + 1) % len(x_train)
    return batch, labels

# define the model


def neural_network_model(data):
    # input data* weights + bias
    hidden_1_layer = {'weights': tf.Variable(tf.random.normal([9, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random.normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random.normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random.normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random.normal([n_classes]))}

    l1 = tf.add(
        tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)  # rectified linear --> activation function

    l2 = tf.add(
        tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(
        tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

# train neural network


def train_neural_network(xin, l):
    prediction = neural_network_model(xin)
    with tf.name_scope('Cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=yin, logits=prediction))
        optimizer = tf.train.AdamOptimizer(
            0.0002).minimize(cost)  # learning rate = 0.001

    tf.summary.scalar("Cost", cost)

    hm_epochs = 20
    eploss = []
    cost_n = []
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(
            "./logs/FertPrediction", sess.graph)  # for 0.8

        sess.run(tf.global_variables_initializer())

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yin, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        merged = tf.summary.merge_all()
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(x_train)/batch_size)):
                epoch_x, epoch_y = generate_batch(batch_size)
                _, c, summary = sess.run([optimizer, cost, merged], feed_dict={
                                         xin: epoch_x, yin: epoch_y})
                cost_n.append(c)
                epoch_loss += c

            writer.add_summary(summary, epoch)
            eploss.append(epoch_loss)

        a = float(accuracy.eval({xin: x_test, yin: y_test}))
        print('accuracy: ', a*100, '%')

        predict = tf.argmax(prediction, 1)
        example = np.array(l)
        example = example.reshape(-1, len(example))
        predict = predict.eval({xin: example})
        print("prediction : Fertilizer",
              label_encoder.inverse_transform(predict))

        return "Fertilizer"+str(label_encoder.inverse_transform(predict))


train_neural_network(xin, [0.7, 0.6, 0.8, 0.8, 0.7, 0.8, 0.2, 0.1, 0.8])
neural_network_model.save("my_model.h5")
print("Model Saved")