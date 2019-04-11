from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.datasets.cifar import load_batch
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export

@tf_export('keras.datasets.cifar10.load_data')
def load_data():
  """Loads CIFAR10 dataset.
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  
  dirname = 'cifar-10-batches-py'
  origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  path = get_file(dirname, origin=origin, untar=True)

  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

def toImage(array, rows = 32, columns = 32):
    return array.reshape(3, rows, columns).transpose([1, 2, 0])

def toData(img, rows = 32, columns = 32):
    return img.transpose([-1, -2, 0]).flatten()

from sklearn.preprocessing import MinMaxScaler
def test_normalize(normalize):
    test_shape = (np.random.choice(range(1000)), 32, 32, 3)
    test_numbers = np.random.choice(range(256), test_shape)
    normalize_out = normalize(test_numbers)

    assert type(normalize_out).__module__ == np.__name__,\
        'Not Numpy Object'

    assert normalize_out.shape == test_shape,\
        'Incorrect Shape. {} shape found'.format(normalize_out.shape)

    assert normalize_out.max() <= 1 and normalize_out.min() >= 0,\
        'Incorect Range. {} to {} found'.format(normalize_out.min(), normalize_out.max())

    _print_success_message()

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    x = x.astype('float32')
    x /= 255.0
    return x

x_train = normalize(x_train)

enc = OneHotEncoder()
y_train = enc.fit_transform(np.array(y_train).reshape(-1,1)).todense()
y_test = enc.transform(np.array(y_test).reshape(-1,1)).todense()


def flipImage(srcImage):
    flippedImages = []
    flippedImages.append(np.fliplr(srcImage))
    flippedImages.append(np.flipud(srcImage))
    flippedImages.append(np.flipud(np.fliplr(srcImage)))
    return flippedImages


def augmentImage(imageVector):
    augmentedImages = []
    image = toImage(imageVector)
    flippedImages = flipImage(image)
    flippedImages.append(image)
    augmentedImages.append(flippedImages)
    
    return augmentedImages

img = augmentImage(x_train[211])

from random import shuffle

def batchIterator(x, y, batchSize, batchCount):
    size = len(x)
    indices = list(range(0, size))
    shuffle(indices)
    indices = indices[0:batchSize * batchCount]
    batches = np.array_split(indices, batchCount)
    for batch in batches:
        yield (x[batch], y[batch])

        
trainX = []
trainY = []

for x, y in zip(x_train, y_train):
    rawAugmentedImages = augmentImage(x)[0]
    trainX.extend(rawAugmentedImages)
    target = [y for i in range(0, len(rawAugmentedImages))]
    trainY.extend(target)

trainX = np.stack(trainX, axis=0)
trainY = np.stack(trainY, axis=0)

processedTestX = []
processedTestY = []

for x, y in zip(x_test, y_test):
    processedTestY.append(y)
    processedTestX.append(toImage(x)) 

processedTestX = np.stack(processedTestX, axis=0)
processedTestY = np.stack(processedTestY, axis=0)

def createConvolutionLayer(inputLayer, kernelHeight, kernelWidth, channelSize, kernelCount, strideX, strideY):
    """This will create a four dimensional tensor
    In this tensor the first and second dimension define the kernel height and width
    The third dimension define the channel size. If the input layer is 
    first layer in neural network then the channel size will be 3 in case of RGB images
    else 1 if images are grey scale. Furthermore if the input layer is Convolution layer 
    then the channel size should be no of kernels in previous layer"""
    
    
    weights = tf.Variable(tf.truncated_normal([kernelHeight, kernelWidth, channelSize, kernelCount], stddev=0.03))
    bias = tf.Variable(tf.constant(0.05, shape=[kernelCount]))
    
    """Stride is also 4 dimensional tensor
    The first and last values should be 1 as they represent the image index and 
    chanel size padding. Second and Third index represent the X and Y strides"""
    layer = tf.nn.conv2d(input = inputLayer, filter = weights, padding='SAME',
                        strides = [1, strideX, strideY, 1]) + bias
    return layer

def flattenLayer(inputLayer):
    """Flatten layer. The first component is image count which is useless"""
    flattenedLayer = tf.reshape(inputLayer, [-1, inputLayer.get_shape()[1:].num_elements()])
    return flattenedLayer

def fullyConnectedLayer(inputLayer, outputLayerCount):
    weights = tf.Variable(tf.truncated_normal(
                        [int(inputLayer.get_shape()[1]), outputLayerCount], stddev=0.03))
    bias = tf.Variable(tf.constant(0.05, shape=[outputLayerCount]))
    layer = tf.matmul(inputLayer, weights) + bias
    return layer

def batchNormalization(inputLayer, isTraining):
    beta = tf.Variable(tf.constant(0.0, shape=[inputLayer.get_shape()[-1]]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[inputLayer.get_shape()[-1]]), name='gamma', trainable=True)
    batchMean, batchVariance = tf.nn.moments(inputLayer, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def meanVarianceUpdate():
        emaOp = ema.apply([batchMean, batchVariance])
        with tf.control_dependencies([emaOp]):
            return tf.identity(batchMean), tf.identity(batchVariance)

    mean, var = tf.cond(isTraining, meanVarianceUpdate, lambda: (ema.average(batchMean), ema.average(batchVariance)))
    normed = tf.nn.batch_normalization(inputLayer, mean, var, beta, gamma, 1e-3)
    return normed


def log_histogram(writer, tag, values, step, bins=1000):
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()
    
"""Input is 4 dimensional tensor -1 so that the no of images can be infered on itself"""
inputLayer = tf.placeholder(tf.float32, [None, 32, 32, 3])
yTrue = tf.placeholder(tf.float32, shape=[None, 10])
isTraining = tf.placeholder(tf.bool, [])


convolutionLayer1 = createConvolutionLayer(inputLayer, 2, 2, 3, 30, 1, 1)
reluActivatedLayer1 = tf.nn.relu(convolutionLayer1)
poolingLayer1 = tf.nn.max_pool(value=convolutionLayer1, ksize=[1, 1, 2, 1], strides = [1, 1, 1, 1], padding='SAME')
dropout0 = tf.nn.dropout(poolingLayer1, tf.to_float(0.5))
bn1 = batchNormalization(dropout0, isTraining)
convolutionLayer2 = createConvolutionLayer(poolingLayer1, 2, 2, 30, 30, 1, 1)
reluActivatedLayer2 = tf.nn.relu(convolutionLayer2)
poolingLayer2 = tf.nn.max_pool(value=convolutionLayer2, ksize=[1, 1, 2, 1], strides = [1, 1, 1, 1], padding='SAME')
dropout = tf.nn.dropout(poolingLayer2, tf.to_float(0.5))
bn2 = batchNormalization(dropout, isTraining)

flattened = flattenLayer(bn2)
fc1 = fullyConnectedLayer(flattened, 950)
reluActivatedLayer2 = tf.nn.relu(fc1)
fc2 = fullyConnectedLayer(flattened, 500)
reluActivatedLayer3 = tf.nn.relu(fc2)
fc= fullyConnectedLayer(reluActivatedLayer3, 10)

predictions = tf.argmax(tf.nn.softmax(fc), axis = 1)
actual = tf.argmax(yTrue, axis = 1)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc, labels = yTrue)

costFunction = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(costFunction)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, actual), tf.float32))

session = tf.Session()
"""Initialize the global variables"""
session.run(tf.global_variables_initializer())

logs_path = '/tmp/tensorflow_logs/exampleCNN/'
summaryWriter =  tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
trainAccList = []
testAccList = []
for i in range(0, 50):
    print("Epoch"+str(i))
    summary = tf.Summary()
    
    for x, y in batchIterator(trainX, trainY, 500, 50):
        session.run(optimizer, feed_dict={inputLayer:x, yTrue:y, isTraining:True})
    
    loss = session.run(costFunction, feed_dict={inputLayer:x, yTrue:y, isTraining:False})
    acc = session.run(accuracy, feed_dict={inputLayer:x, yTrue:y, isTraining:False})    
    summary.value.add(tag = "TrainingLoss", simple_value = loss)
    summary.value.add(tag = "TrainingAcc", simple_value = acc)
    trainAccList.append(acc)
    print("Training Accuracy: ", trainAccList)
        
    lossTestList = []
    accTestList = []
    for x, y in batchIterator(processedTestX, processedTestY, 1000, 5):
        lossTest = session.run(costFunction, feed_dict={inputLayer:x, yTrue:y, isTraining:False})
        accTest = session.run(accuracy, feed_dict={inputLayer:x, yTrue:y, isTraining:False})
        lossTestList.append(lossTest)
        accTestList.append(accTest)
    summary.value.add(tag = "TestLoss", simple_value = np.mean(lossTestList))
    summary.value.add(tag = "TestAcc", simple_value = np.mean(accTestList))
    testAccList.append(np.mean(accTestList))
    print("Testing Accuracy: ", testAccList)
    summaryWriter.add_summary(summary, i)
    

log_histogram(summaryWriter, "TrainAccHist", trainAccList, 50)
log_histogram(summaryWriter, "TestAccHist", testAccList, 50)
session.close()
