# Import
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# convert an array of values into a dataset matrix
def create_dataset(dataset):
  dataX, dataY = [], []
  for i in range(len(dataset)-1):
    dataX.append(dataset[i])
    dataY.append(dataset[i + 1])
  return np.asarray(dataX), np.asarray(dataY)

# Import data
# data = pd.read_csv('eth.csv')
data = pd.read_csv('btc.csv')

# Drop date variable
data = data.drop(['DATE'], 1)
data = data.drop(['market_cap'], 1)

# Make data a np.array
data = data.values

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

#prepare the X and Y label
X,y = create_dataset(data)

#Take 80% of data as the training sample and 20% as testing sample
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# Number of stocks in training data
print("X_train cols", X_train.shape[1])
print("X_train rows", X_train.shape[0])
print("X_test cols", X_test.shape[1])
print("X_test rows", X_test.shape[0])
print("y_train cols", y_train.shape[1])
print("y_train rows", y_train.shape[0])
print("y_test cols", y_test.shape[1])
print("y_test rows", y_test.shape[0])

# Neurons
n_neurons_1 = 2048
n_neurons_2 = 1024
n_neurons_3 = 512
n_neurons_4 = 256

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([X_train.shape[1], n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test)
plt.show()

# Fit neural net
batch_size = 64
mse_train = []
mse_test = []

# Run
epochs = 10
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        # if np.mod(i, 50) == 0:

        # MSE train and test
        mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
        mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
        print('MSE Train: ', mse_train[-1])
        print('MSE Test: ', mse_test[-1])

        # Prediction
        pred = net.run(out, feed_dict={X: X_test})
        line2.set_ydata(pred)
        plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
        plt.pause(.01)