### Deep Learning Foundations Nanodegree
# Project: Image Classification

---
# Results
The sections below outline the work I completed as part of this project. The Jupyter Notebook document containing the source code is located [here](https://github.com/tommytracey/udacity/blob/master/deep-learning-nano/projects/2-image-classification/dlnd_image_classification-v2.ipynb).

## Overview
In this project, we classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  The dataset consists of airplanes, dogs, cats, and other objects. You'll preprocess the images, then train a convolutional neural network on all the samples. The images need to be normalized and the labels need to be one-hot encoded.  We get to apply what you learned and build a convolutional, max pooling, dropout, and fully connected layers.  At the end, we get to see your neural network's predictions on the sample images.

## Get the Data
Run the following cell to download the [CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)
```

    All files found!


## Explore the Data
The dataset is broken into batches to prevent your machine from running out of memory.  The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Understanding a dataset is part of making predictions on the data.  Play around with the code cell below by changing the `batch_id` and `sample_id`. The `batch_id` is the id for a batch (1-5). The `sample_id` is the id for a image and label pair in the batch.

Ask yourself "What are all possible labels?", "What is the range of values for the image data?", "Are the labels in order or random?".  Answers to questions like these will help you preprocess the data and end up with better predictions.


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np

# Explore the dataset
batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```


    Stats of batch 1:
    Samples: 10000
    Label Counts: {0: 1005, 1: 974, 2: 1032, 3: 1016, 4: 999, 5: 937, 6: 1030, 7: 1001, 8: 1025, 9: 981}
    First 20 Labels: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7, 2, 9, 9, 9, 3, 2, 6]

    Example of Image 5:
    Image - Min Value: 0 Max Value: 252
    Image - Shape: (32, 32, 3)
    Label - Label Id: 1 Name: automobile



![png](output_3_1.png)


## Implement Preprocess Functions
### Normalize
In the cell below, implement the `normalize` function to take in image data, `x`, and return it as a normalized Numpy array. The values should be in the range of 0 to 1, inclusive.  The return object should be the same shape as `x`.


```python
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    a = 0
    b = 1
    x_min = 0
    x_max = 255
    return x / x_max


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)
```

    Tests Passed


### One-hot encode
Just like the previous code cell, you'll be implementing a function for preprocessing.  This time, you'll implement the `one_hot_encode` function. The input, `x`, are a list of labels.  Implement the function to return the list of labels as One-Hot encoded Numpy array.  The possible values for labels are 0 to 9. The one-hot encoding function should return the same encoding for each value between each call to `one_hot_encode`.  Make sure to save the map of encodings outside the function.

Hint: Don't reinvent the wheel.


```python
# from sklearn import preprocessing

# lbft = preprocessing.LabelBinarizer().fit_transform([i for i in range(10)])

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # return np.array([lbft[i] for i in x])
    return np.eye(10)[x]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)
```

    Tests Passed


### Randomize Data
As you saw from exploring the data above, the order of the samples are randomized.  It doesn't hurt to randomize it again, but you don't need to for this dataset.

## Preprocess all the data and save it
Running the code cell below will preprocess all the CIFAR-10 data and save it to file. The code below also uses 10% of the training data for validation.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
```

# Check Point
This is your first checkpoint.  If you ever decide to come back to this notebook or have to restart the notebook, you can start from here.  The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
```

## Build the network
For the neural network, you'll build each layer into a function.  Most of the code you've seen has been outside of functions. To test your code more thoroughly, we require that you put each layer in a function.  This allows us to give you better feedback and test for simple mistakes using our unittests before you submit your project.

>**Note:** If you're finding it hard to dedicate enough time for this course each week, we've provided a small shortcut to this part of the project. In the next couple of problems, you'll have the option to use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages to build each layer, except the layers you build in the "Convolutional and Max Pooling Layer" section.  TF Layers is similar to Keras's and TFLearn's abstraction to layers, so it's easy to pickup.

>However, if you would like to get the most out of this course, try to solve all the problems _without_ using anything from the TF Layers packages. You **can** still use classes from other packages that happen to have the same name as ones you find in TF Layers! For example, instead of using the TF Layers version of the `conv2d` class, [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d), you would want to use the TF Neural Network version of `conv2d`, [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d).

Let's begin!

### Input
The neural network needs to read the image data, one-hot encoded labels, and dropout keep probability. Implement the following functions
* Implement `neural_net_image_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `image_shape` with batch size set to `None`.
 * Name the TensorFlow placeholder "x" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_label_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `n_classes` with batch size set to `None`.
 * Name the TensorFlow placeholder "y" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_keep_prob_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) for dropout keep probability.
 * Name the TensorFlow placeholder "keep_prob" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).

These names will be used at the end of the project to load your saved model.

Note: `None` for shapes in TensorFlow allow for a dynamic size.


```python
import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, shape=[None, *image_shape], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name='keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
```

    Image Input Tests Passed.
    Label Input Tests Passed.
    Keep Prob Tests Passed.


### Convolution and Max Pooling Layer
Convolution layers have a lot of success with images. For this code cell, you should implement the function `conv2d_maxpool` to apply convolution then max pooling:
* Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
* Apply a convolution to `x_tensor` using weight and `conv_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.
* Add bias
* Add a nonlinear activation to the convolution.
* Apply Max Pooling using `pool_ksize` and `pool_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.

**Note:** You **can't** use [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) for **this** layer, but you can still use TensorFlow's [Neural Network](https://www.tensorflow.org/api_docs/python/tf/nn) package. You may still use the shortcut option for all the **other** layers.


```python
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # Weight and bias
    x_depth = x_tensor.get_shape().as_list()[3]

    weight = tf.Variable(tf.truncated_normal([*conv_ksize, x_depth, conv_num_outputs],\
                                          dtype=tf.float32, mean=0.0, stddev=0.1))
    bias = tf.Variable(tf.zeros(conv_num_outputs))

    # Apply convolution, bias, and non-linear activation
    conv_strides_list = [1, conv_strides[0], conv_strides[1], 1]

    conv_layer = tf.nn.conv2d(tf.to_float(x_tensor), weight, strides=conv_strides_list, padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    conv_layer = tf.nn.relu(conv_layer)

    # Apply max pooling
    pool_ksize_list = [1, pool_ksize[0], pool_ksize[1], 1]
    pool_strides_list = [1, pool_strides[0], pool_strides[1], 1]

    return tf.nn.max_pool(conv_layer, pool_ksize_list, pool_strides_list, padding='SAME')



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)
```

    Tests Passed


### Flatten Layer
Implement the `flatten` function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor.  The output should be the shape (*Batch Size*, *Flattened Image Size*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    shape = x_tensor.get_shape().as_list()
    dim = shape[1] * shape[2] * shape[3]
    return tf.reshape(x_tensor, [-1, dim])


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)
```

    Tests Passed


### Fully-Connected Layer
Implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    conv_inputs = x_tensor.get_shape().as_list()[1]
    weight = tf.Variable(tf.truncated_normal([conv_inputs, num_outputs], mean=0.0, stddev=0.01))
    bias = tf.Variable(tf.zeros(num_outputs))

    fc_layer = tf.add(tf.matmul(x_tensor, weight), bias)
    return tf.nn.relu(fc_layer)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)
```

    Tests Passed


### Output Layer
Implement the `output` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

**Note:** Activation, softmax, or cross entropy should **not** be applied to this.


```python
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    fc_inputs = x_tensor.get_shape().as_list()[1]
    weight = tf.Variable(tf.random_normal([fc_inputs, num_outputs], mean=0.0, stddev=0.01))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul (x_tensor, weight), bias)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)
```

    Tests Passed


### Create Convolutional Model
Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits.  Use the layers you created above to create this model:

* Apply 1, 2, or 3 Convolution and Max Pool layers
* Apply a Flatten Layer
* Apply 1, 2, or 3 Fully Connected Layers
* Apply an Output Layer
* Return the output
* Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`.


```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    x_tensor = x

    conv_num_outputs1 = 30
    conv_num_outputs2 = 90
    conv_num_outputs3 = 180

    conv_ksize = (4,4)
    conv_strides = (1,1)
    pool_ksize = (2,2)
    pool_strides = (2,2)
    num_outputs = 500 # fc layer

    # Convolutional layers
    conv = conv2d_maxpool(x_tensor, conv_num_outputs1, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv = conv2d_maxpool(conv, conv_num_outputs2, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv = conv2d_maxpool(conv, conv_num_outputs3, conv_ksize, conv_strides, pool_ksize, pool_strides)

    # Flatten layer
    flat = flatten(conv)

    # Fully connected layer(s)
    fc = fully_conn(flat, num_outputs)
    fc = tf.nn.dropout(fc, tf.to_float(keep_prob))

    # Output layer
    out = output(fc, 10)

    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)
```

    Neural Network Built!


## Train the Neural Network
### Single Optimization
Implement the function `train_neural_network` to do a single optimization.  The optimization should use `optimizer` to optimize in `session` with a `feed_dict` of the following:
* `x` for image input
* `y` for labels
* `keep_prob` for keep probability for dropout

This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.

Note: Nothing needs to be returned. This function is only optimizing the neural network.


```python
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})

    pass

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)
```

    Tests Passed


### Show Stats
Implement the function `print_stats` to print loss and validation accuracy.  Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy.  Use a keep probability of `1.0` to calculate the loss and validation accuracy.


```python
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    current_cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})

    valid_accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})

    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch,
        current_cost,
        valid_accuracy))
```

### Hyperparameters
Tune the following parameters:
* Set `epochs` to the number of iterations until the network stops learning or start overfitting
* Set `batch_size` to the highest number that your machine has memory for.  Most people set them to common sizes of memory:
 * 64
 * 128
 * 256
 * ...
* Set `keep_probability` to the probability of keeping a node using dropout


```python
# TODO: Tune Parameters
epochs = 25
batch_size = 128
keep_probability = .6
```

### Train on a Single CIFAR-10 Batch
Instead of training the neural network on all the CIFAR-10 batches of data, let's use a single batch. This should save time while you iterate on the model to get a better accuracy.  Once the final validation accuracy is 50% or greater, run the model on all the data in the next section.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
```

    Checking the Training on a Single Batch...
    Epoch  1, CIFAR-10 Batch 1:  Epoch: 0    - Cost: 1.95     Valid Accuracy: 0.378
    Epoch  2, CIFAR-10 Batch 1:  Epoch: 1    - Cost: 1.65     Valid Accuracy: 0.447
    Epoch  3, CIFAR-10 Batch 1:  Epoch: 2    - Cost: 1.36     Valid Accuracy: 0.485
    Epoch  4, CIFAR-10 Batch 1:  Epoch: 3    - Cost: 1.17     Valid Accuracy: 0.535
    Epoch  5, CIFAR-10 Batch 1:  Epoch: 4    - Cost: 0.921    Valid Accuracy: 0.563
    Epoch  6, CIFAR-10 Batch 1:  Epoch: 5    - Cost: 0.718    Valid Accuracy: 0.579
    Epoch  7, CIFAR-10 Batch 1:  Epoch: 6    - Cost: 0.494    Valid Accuracy: 0.595
    Epoch  8, CIFAR-10 Batch 1:  Epoch: 7    - Cost: 0.363    Valid Accuracy: 0.588
    Epoch  9, CIFAR-10 Batch 1:  Epoch: 8    - Cost: 0.277    Valid Accuracy: 0.602
    Epoch 10, CIFAR-10 Batch 1:  Epoch: 9    - Cost: 0.242    Valid Accuracy: 0.599
    Epoch 11, CIFAR-10 Batch 1:  Epoch: 10   - Cost: 0.131    Valid Accuracy: 0.616
    Epoch 12, CIFAR-10 Batch 1:  Epoch: 11   - Cost: 0.0957   Valid Accuracy: 0.616
    Epoch 13, CIFAR-10 Batch 1:  Epoch: 12   - Cost: 0.087    Valid Accuracy: 0.614
    Epoch 14, CIFAR-10 Batch 1:  Epoch: 13   - Cost: 0.0727   Valid Accuracy: 0.608
    Epoch 15, CIFAR-10 Batch 1:  Epoch: 14   - Cost: 0.074    Valid Accuracy: 0.588
    Epoch 16, CIFAR-10 Batch 1:  Epoch: 15   - Cost: 0.0462   Valid Accuracy: 0.613
    Epoch 17, CIFAR-10 Batch 1:  Epoch: 16   - Cost: 0.0415   Valid Accuracy: 0.606
    Epoch 18, CIFAR-10 Batch 1:  Epoch: 17   - Cost: 0.0181   Valid Accuracy: 0.611
    Epoch 19, CIFAR-10 Batch 1:  Epoch: 18   - Cost: 0.018    Valid Accuracy: 0.605
    Epoch 20, CIFAR-10 Batch 1:  Epoch: 19   - Cost: 0.011    Valid Accuracy: 0.611
    Epoch 21, CIFAR-10 Batch 1:  Epoch: 20   - Cost: 0.00917  Valid Accuracy: 0.618
    Epoch 22, CIFAR-10 Batch 1:  Epoch: 21   - Cost: 0.014    Valid Accuracy: 0.601
    Epoch 23, CIFAR-10 Batch 1:  Epoch: 22   - Cost: 0.012    Valid Accuracy: 0.589
    Epoch 24, CIFAR-10 Batch 1:  Epoch: 23   - Cost: 0.00581  Valid Accuracy: 0.613
    Epoch 25, CIFAR-10 Batch 1:  Epoch: 24   - Cost: 0.0055   Valid Accuracy: 0.614


### Fully Train the Model
Now that you got a good accuracy with a single CIFAR-10 batch, try it with all five batches.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
```

    Training...
    Epoch  1, CIFAR-10 Batch 1:  Epoch: 0    - Cost: 1.98     Valid Accuracy: 0.378
    Epoch  1, CIFAR-10 Batch 2:  Epoch: 0    - Cost: 1.56     Valid Accuracy: 0.41
    Epoch  1, CIFAR-10 Batch 3:  Epoch: 0    - Cost: 1.26     Valid Accuracy: 0.496
    Epoch  1, CIFAR-10 Batch 4:  Epoch: 0    - Cost: 1.27     Valid Accuracy: 0.522
    Epoch  1, CIFAR-10 Batch 5:  Epoch: 0    - Cost: 1.23     Valid Accuracy: 0.583
    Epoch  2, CIFAR-10 Batch 1:  Epoch: 1    - Cost: 1.23     Valid Accuracy: 0.548
    Epoch  2, CIFAR-10 Batch 2:  Epoch: 1    - Cost: 0.945    Valid Accuracy: 0.587
    Epoch  2, CIFAR-10 Batch 3:  Epoch: 1    - Cost: 0.821    Valid Accuracy: 0.607
    Epoch  2, CIFAR-10 Batch 4:  Epoch: 1    - Cost: 0.904    Valid Accuracy: 0.642
    Epoch  2, CIFAR-10 Batch 5:  Epoch: 1    - Cost: 0.75     Valid Accuracy: 0.654
    Epoch  3, CIFAR-10 Batch 1:  Epoch: 2    - Cost: 0.888    Valid Accuracy: 0.651
    Epoch  3, CIFAR-10 Batch 2:  Epoch: 2    - Cost: 0.639    Valid Accuracy: 0.664
    Epoch  3, CIFAR-10 Batch 3:  Epoch: 2    - Cost: 0.458    Valid Accuracy: 0.666
    Epoch  3, CIFAR-10 Batch 4:  Epoch: 2    - Cost: 0.548    Valid Accuracy: 0.685
    Epoch  3, CIFAR-10 Batch 5:  Epoch: 2    - Cost: 0.469    Valid Accuracy: 0.682
    Epoch  4, CIFAR-10 Batch 1:  Epoch: 3    - Cost: 0.627    Valid Accuracy: 0.682
    Epoch  4, CIFAR-10 Batch 2:  Epoch: 3    - Cost: 0.48     Valid Accuracy: 0.679
    Epoch  4, CIFAR-10 Batch 3:  Epoch: 3    - Cost: 0.333    Valid Accuracy: 0.689
    Epoch  4, CIFAR-10 Batch 4:  Epoch: 3    - Cost: 0.347    Valid Accuracy: 0.711
    Epoch  4, CIFAR-10 Batch 5:  Epoch: 3    - Cost: 0.315    Valid Accuracy: 0.689
    Epoch  5, CIFAR-10 Batch 1:  Epoch: 4    - Cost: 0.393    Valid Accuracy: 0.712
    Epoch  5, CIFAR-10 Batch 2:  Epoch: 4    - Cost: 0.349    Valid Accuracy: 0.694
    Epoch  5, CIFAR-10 Batch 3:  Epoch: 4    - Cost: 0.236    Valid Accuracy: 0.716
    Epoch  5, CIFAR-10 Batch 4:  Epoch: 4    - Cost: 0.185    Valid Accuracy: 0.715
    Epoch  5, CIFAR-10 Batch 5:  Epoch: 4    - Cost: 0.224    Valid Accuracy: 0.716
    Epoch  6, CIFAR-10 Batch 1:  Epoch: 5    - Cost: 0.242    Valid Accuracy: 0.711
    Epoch  6, CIFAR-10 Batch 2:  Epoch: 5    - Cost: 0.249    Valid Accuracy: 0.704
    Epoch  6, CIFAR-10 Batch 3:  Epoch: 5    - Cost: 0.175    Valid Accuracy: 0.711
    Epoch  6, CIFAR-10 Batch 4:  Epoch: 5    - Cost: 0.182    Valid Accuracy: 0.717
    Epoch  6, CIFAR-10 Batch 5:  Epoch: 5    - Cost: 0.147    Valid Accuracy: 0.705
    Epoch  7, CIFAR-10 Batch 1:  Epoch: 6    - Cost: 0.151    Valid Accuracy: 0.716
    Epoch  7, CIFAR-10 Batch 2:  Epoch: 6    - Cost: 0.145    Valid Accuracy: 0.699
    Epoch  7, CIFAR-10 Batch 3:  Epoch: 6    - Cost: 0.114    Valid Accuracy: 0.706
    Epoch  7, CIFAR-10 Batch 4:  Epoch: 6    - Cost: 0.167    Valid Accuracy: 0.705
    Epoch  7, CIFAR-10 Batch 5:  Epoch: 6    - Cost: 0.103    Valid Accuracy: 0.718
    Epoch  8, CIFAR-10 Batch 1:  Epoch: 7    - Cost: 0.092    Valid Accuracy: 0.714
    Epoch  8, CIFAR-10 Batch 2:  Epoch: 7    - Cost: 0.173    Valid Accuracy: 0.718
    Epoch  8, CIFAR-10 Batch 3:  Epoch: 7    - Cost: 0.0779   Valid Accuracy: 0.711
    Epoch  8, CIFAR-10 Batch 4:  Epoch: 7    - Cost: 0.109    Valid Accuracy: 0.724
    Epoch  8, CIFAR-10 Batch 5:  Epoch: 7    - Cost: 0.0632   Valid Accuracy: 0.724
    Epoch  9, CIFAR-10 Batch 1:  Epoch: 8    - Cost: 0.0653   Valid Accuracy: 0.723
    Epoch  9, CIFAR-10 Batch 2:  Epoch: 8    - Cost: 0.116    Valid Accuracy: 0.709
    Epoch  9, CIFAR-10 Batch 3:  Epoch: 8    - Cost: 0.0725   Valid Accuracy: 0.706
    Epoch  9, CIFAR-10 Batch 4:  Epoch: 8    - Cost: 0.0627   Valid Accuracy: 0.721
    Epoch  9, CIFAR-10 Batch 5:  Epoch: 8    - Cost: 0.0502   Valid Accuracy: 0.728
    Epoch 10, CIFAR-10 Batch 1:  Epoch: 9    - Cost: 0.0638   Valid Accuracy: 0.724
    Epoch 10, CIFAR-10 Batch 2:  Epoch: 9    - Cost: 0.0678   Valid Accuracy: 0.718
    Epoch 10, CIFAR-10 Batch 3:  Epoch: 9    - Cost: 0.0323   Valid Accuracy: 0.736
    Epoch 10, CIFAR-10 Batch 4:  Epoch: 9    - Cost: 0.0357   Valid Accuracy: 0.707
    Epoch 10, CIFAR-10 Batch 5:  Epoch: 9    - Cost: 0.0548   Valid Accuracy: 0.726
    Epoch 11, CIFAR-10 Batch 1:  Epoch: 10   - Cost: 0.0467   Valid Accuracy: 0.726
    Epoch 11, CIFAR-10 Batch 2:  Epoch: 10   - Cost: 0.0414   Valid Accuracy: 0.715
    Epoch 11, CIFAR-10 Batch 3:  Epoch: 10   - Cost: 0.0648   Valid Accuracy: 0.717
    Epoch 11, CIFAR-10 Batch 4:  Epoch: 10   - Cost: 0.0435   Valid Accuracy: 0.704
    Epoch 11, CIFAR-10 Batch 5:  Epoch: 10   - Cost: 0.0205   Valid Accuracy: 0.723
    Epoch 12, CIFAR-10 Batch 1:  Epoch: 11   - Cost: 0.0271   Valid Accuracy: 0.719
    Epoch 12, CIFAR-10 Batch 2:  Epoch: 11   - Cost: 0.0342   Valid Accuracy: 0.714
    Epoch 12, CIFAR-10 Batch 3:  Epoch: 11   - Cost: 0.0308   Valid Accuracy: 0.712
    Epoch 12, CIFAR-10 Batch 4:  Epoch: 11   - Cost: 0.0254   Valid Accuracy: 0.719
    Epoch 12, CIFAR-10 Batch 5:  Epoch: 11   - Cost: 0.0189   Valid Accuracy: 0.726
    Epoch 13, CIFAR-10 Batch 1:  Epoch: 12   - Cost: 0.02     Valid Accuracy: 0.724
    Epoch 13, CIFAR-10 Batch 2:  Epoch: 12   - Cost: 0.024    Valid Accuracy: 0.718
    Epoch 13, CIFAR-10 Batch 3:  Epoch: 12   - Cost: 0.0109   Valid Accuracy: 0.721
    Epoch 13, CIFAR-10 Batch 4:  Epoch: 12   - Cost: 0.0248   Valid Accuracy: 0.707
    Epoch 13, CIFAR-10 Batch 5:  Epoch: 12   - Cost: 0.0104   Valid Accuracy: 0.731
    Epoch 14, CIFAR-10 Batch 1:  Epoch: 13   - Cost: 0.0512   Valid Accuracy: 0.714
    Epoch 14, CIFAR-10 Batch 2:  Epoch: 13   - Cost: 0.0164   Valid Accuracy: 0.711
    Epoch 14, CIFAR-10 Batch 3:  Epoch: 13   - Cost: 0.0148   Valid Accuracy: 0.718
    Epoch 14, CIFAR-10 Batch 4:  Epoch: 13   - Cost: 0.0235   Valid Accuracy: 0.716
    Epoch 14, CIFAR-10 Batch 5:  Epoch: 13   - Cost: 0.0116   Valid Accuracy: 0.712
    Epoch 15, CIFAR-10 Batch 1:  Epoch: 14   - Cost: 0.0174   Valid Accuracy: 0.723
    Epoch 15, CIFAR-10 Batch 2:  Epoch: 14   - Cost: 0.00721  Valid Accuracy: 0.701
    Epoch 15, CIFAR-10 Batch 3:  Epoch: 14   - Cost: 0.00424  Valid Accuracy: 0.715
    Epoch 15, CIFAR-10 Batch 4:  Epoch: 14   - Cost: 0.0123   Valid Accuracy: 0.722
    Epoch 15, CIFAR-10 Batch 5:  Epoch: 14   - Cost: 0.00479  Valid Accuracy: 0.71
    Epoch 16, CIFAR-10 Batch 1:  Epoch: 15   - Cost: 0.00379  Valid Accuracy: 0.73
    Epoch 16, CIFAR-10 Batch 2:  Epoch: 15   - Cost: 0.00291  Valid Accuracy: 0.703
    Epoch 16, CIFAR-10 Batch 3:  Epoch: 15   - Cost: 0.0135   Valid Accuracy: 0.717
    Epoch 16, CIFAR-10 Batch 4:  Epoch: 15   - Cost: 0.0205   Valid Accuracy: 0.724
    Epoch 16, CIFAR-10 Batch 5:  Epoch: 15   - Cost: 0.0108   Valid Accuracy: 0.702
    Epoch 17, CIFAR-10 Batch 1:  Epoch: 16   - Cost: 0.00534  Valid Accuracy: 0.731
    Epoch 17, CIFAR-10 Batch 2:  Epoch: 16   - Cost: 0.00541  Valid Accuracy: 0.726
    Epoch 17, CIFAR-10 Batch 3:  Epoch: 16   - Cost: 0.0134   Valid Accuracy: 0.728
    Epoch 17, CIFAR-10 Batch 4:  Epoch: 16   - Cost: 0.00776  Valid Accuracy: 0.728
    Epoch 17, CIFAR-10 Batch 5:  Epoch: 16   - Cost: 0.00274  Valid Accuracy: 0.7  
    Epoch 18, CIFAR-10 Batch 1:  Epoch: 17   - Cost: 0.0171   Valid Accuracy: 0.71
    Epoch 18, CIFAR-10 Batch 2:  Epoch: 17   - Cost: 0.00649  Valid Accuracy: 0.711
    Epoch 18, CIFAR-10 Batch 3:  Epoch: 17   - Cost: 0.00477  Valid Accuracy: 0.731
    Epoch 18, CIFAR-10 Batch 4:  Epoch: 17   - Cost: 0.0057   Valid Accuracy: 0.718
    Epoch 18, CIFAR-10 Batch 5:  Epoch: 17   - Cost: 0.00183  Valid Accuracy: 0.718
    Epoch 19, CIFAR-10 Batch 1:  Epoch: 18   - Cost: 0.0109   Valid Accuracy: 0.708
    Epoch 19, CIFAR-10 Batch 2:  Epoch: 18   - Cost: 0.00252  Valid Accuracy: 0.715
    Epoch 19, CIFAR-10 Batch 3:  Epoch: 18   - Cost: 0.00328  Valid Accuracy: 0.727
    Epoch 19, CIFAR-10 Batch 4:  Epoch: 18   - Cost: 0.0253   Valid Accuracy: 0.714
    Epoch 19, CIFAR-10 Batch 5:  Epoch: 18   - Cost: 0.0144   Valid Accuracy: 0.711
    Epoch 20, CIFAR-10 Batch 1:  Epoch: 19   - Cost: 0.0082   Valid Accuracy: 0.711
    Epoch 20, CIFAR-10 Batch 2:  Epoch: 19   - Cost: 0.00129  Valid Accuracy: 0.731
    Epoch 20, CIFAR-10 Batch 3:  Epoch: 19   - Cost: 0.00225  Valid Accuracy: 0.733
    Epoch 20, CIFAR-10 Batch 4:  Epoch: 19   - Cost: 0.00617  Valid Accuracy: 0.715
    Epoch 20, CIFAR-10 Batch 5:  Epoch: 19   - Cost: 0.00116  Valid Accuracy: 0.729
    Epoch 21, CIFAR-10 Batch 1:  Epoch: 20   - Cost: 0.00343  Valid Accuracy: 0.719
    Epoch 21, CIFAR-10 Batch 2:  Epoch: 20   - Cost: 0.00327  Valid Accuracy: 0.715
    Epoch 21, CIFAR-10 Batch 3:  Epoch: 20   - Cost: 0.00158  Valid Accuracy: 0.725
    Epoch 21, CIFAR-10 Batch 4:  Epoch: 20   - Cost: 0.0203   Valid Accuracy: 0.719
    Epoch 21, CIFAR-10 Batch 5:  Epoch: 20   - Cost: 0.00124  Valid Accuracy: 0.724
    Epoch 22, CIFAR-10 Batch 1:  Epoch: 21   - Cost: 0.00201  Valid Accuracy: 0.717
    Epoch 22, CIFAR-10 Batch 2:  Epoch: 21   - Cost: 0.0047   Valid Accuracy: 0.719
    Epoch 22, CIFAR-10 Batch 3:  Epoch: 21   - Cost: 0.00316  Valid Accuracy: 0.733
    Epoch 22, CIFAR-10 Batch 4:  Epoch: 21   - Cost: 0.00118  Valid Accuracy: 0.728
    Epoch 22, CIFAR-10 Batch 5:  Epoch: 21   - Cost: 0.00161  Valid Accuracy: 0.732
    Epoch 23, CIFAR-10 Batch 1:  Epoch: 22   - Cost: 0.00935  Valid Accuracy: 0.724
    Epoch 23, CIFAR-10 Batch 2:  Epoch: 22   - Cost: 0.00319  Valid Accuracy: 0.725
    Epoch 23, CIFAR-10 Batch 3:  Epoch: 22   - Cost: 0.00327  Valid Accuracy: 0.737
    Epoch 23, CIFAR-10 Batch 4:  Epoch: 22   - Cost: 0.00464  Valid Accuracy: 0.72
    Epoch 23, CIFAR-10 Batch 5:  Epoch: 22   - Cost: 0.00265  Valid Accuracy: 0.723
    Epoch 24, CIFAR-10 Batch 1:  Epoch: 23   - Cost: 0.00247  Valid Accuracy: 0.716
    Epoch 24, CIFAR-10 Batch 2:  Epoch: 23   - Cost: 0.0101   Valid Accuracy: 0.719
    Epoch 24, CIFAR-10 Batch 3:  Epoch: 23   - Cost: 0.00361  Valid Accuracy: 0.732
    Epoch 24, CIFAR-10 Batch 4:  Epoch: 23   - Cost: 0.00338  Valid Accuracy: 0.714
    Epoch 24, CIFAR-10 Batch 5:  Epoch: 23   - Cost: 0.00172  Valid Accuracy: 0.727
    Epoch 25, CIFAR-10 Batch 1:  Epoch: 24   - Cost: 0.000594 Valid Accuracy: 0.717
    Epoch 25, CIFAR-10 Batch 2:  Epoch: 24   - Cost: 0.0014   Valid Accuracy: 0.719
    Epoch 25, CIFAR-10 Batch 3:  Epoch: 24   - Cost: 0.00395  Valid Accuracy: 0.733
    Epoch 25, CIFAR-10 Batch 4:  Epoch: 24   - Cost: 0.0165   Valid Accuracy: 0.719
    Epoch 25, CIFAR-10 Batch 5:  Epoch: 24   - Cost: 0.000224 Valid Accuracy: 0.724


# Checkpoint
The model has been saved to disk.
## Test Model
Test your model against the test dataset.  This will be your final accuracy. You should have an accuracy greater than 50%. If you don't, keep tweaking the model architecture and parameters.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
```

    Testing Accuracy: 0.7212223101265823




![png](output_36_1.png)


## Why 50-70% Accuracy?
You might be wondering why you can't get an accuracy any higher. First things first, 50% isn't bad for a simple CNN.  Pure guessing would get you 10% accuracy. However, you might notice people are getting scores [well above 70%](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).  That's because we haven't taught you all there is to know about neural networks. We still need to cover a few more techniques.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook.  Save the notebook file as "dlnd_image_classification.ipynb" and save it as a HTML file under "File" -> "Download as".  Include the "helper.py" and "problem_unittests.py" files in your submission.
