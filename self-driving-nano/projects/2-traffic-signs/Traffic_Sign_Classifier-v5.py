
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# In[59]:

## global variables

LOGDIR = ".log_dir"
IS_TRAIN_PHASE = tf.placeholder(dtype=tf.bool, name='is_train_phase')


# In[21]:

## imports



# ---
# ## Step 0: Load The Data

# In[3]:

# Load pickled data
import pickle

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[4]:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

import numpy as np

# TODO: Number of training examples
n_train = len(x_train)

# TODO: Number of validation examples
n_valid = len(x_valid)

# TODO: Number of testing examples.
n_test = len(x_test)

# TODO: What's the shape of a traffic sign image?
image_shape = x_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[47]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import matplotlib.pyplot as plt

# Show visualizations in the notebook
get_ipython().magic('matplotlib inline')

index = random.randint(0, len(x_train))
image = x_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[15]:

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle

x_train, y_train = shuffle(x_train, y_train)


# **My approach:**
# - normalization will be handled within the network
# - data augmentation will include grayscaling and affine transformations

# ### Data Augmentation

# In[ ]:




# ### Model Architecture

# In[ ]:

### Define your architecture here.
### Feel free to use as many code cells as needed.


# In[80]:

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

learning_rate = 0.001

mu = 0  # normalized mean
sigma = 0.1  # normalized stdev


# #### Operations

# The series of functions below are designed to make the model more modular. This reduces the amount of hard-coding and makes it much easier to experiment with different model architectures.

# In[100]:

# get weights
def get_weights(shape):
#     H, W, C, K = shape
    w = tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma), name="W")
    return w


# get biases
def get_biases(shape):
    b = tf.Variable(tf.constant(0.01, shape=[shape]), name="B")
    return b


# create convolutional layer
def create_conv(input, n_kernels=1, kernel_size=(1, 1), strides=[1, 1, 1, 1], name="conv"):
    H, W = kernel_size        # filter height, width
    C = input.get_shape().as_list()[3]  # input depth
    K = n_kernels             # output depth
    shape = [H, W, C, K]
    print('shape: ', shape)
    
    with tf.name_scope(name):
        w = get_weights(shape)
        b = get_biases(K)
        conv = tf.nn.conv2d(input, w, strides=strides, padding='SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


# create fully connected layer
def create_fc(input, n_inputs, n_outputs, name="fc"):
    with tf.name_scope(name):    
        w = tf.Variable(tf.truncated_normal([n_inputs, n_outputs], mean=mu, stddev=sigma), name="W")
        b = tf.Variable(tf.constant(0.01, shape=[n_outputs]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


# apply batch normalization
def batch_norm(input, decay=0.9, eps=1e-5, is_training=1, reuse=None, name="bn"):
    with tf.name_scope(name):
        bn = tf.contrib.layers.batch_norm(input, decay=decay, epsilon=eps, center=True, scale=True,
                              is_training=is_training, reuse=reuse, # for testing set is_training=0, reuse=True
                              updates_collections=None, scope=scope) 
        return bn

               
# apply relu
def relu(input, name="relu"):
    with tf.name_scope(name):
        act = tf.nn.relu(input, name=name)
        tf.summary.histogram("relu", act)        
        return act

    
# apply max pooling
def max_pool(input, kernel_size=(1,1), strides=[1,1,1,1], name="maxpool"):
    H = kernel_size[0]
    W = kernel_size[1]
    
    with tf.name_scope(name):    
        pool = tf.nn.max_pool(input, ksize=[1, H, W, 1], strides=strides, padding='SAME', name=name)
        return pool


# apply avg pooling
def avg_pool(input, name="avgpool"):
    H, W = input.get_shape()[1:2]
    
    with tf.name_scope(name):    
        pool = tf.nn.avg_pool(input, ksize=[1,H,W,1], strides=[1,H,W,1], padding='VALID', name=name)
        pool = flatten(pool)
        return pool


# apply dropout
def dropout(input, keep=1.0, name="drop"):
    with tf.name_scope(name):    
        drop = tf.cond(IS_TRAIN_PHASE,
                       lambda: tf.nn.dropout(input, keep),
                       lambda: tf.nn.dropout(input, 1))
        return drop


# flatten layer
def flatten(inputs, name="flat"):
    with tf.name_scope(name):    
        # returns a flattened tensor with shape [batch_size, k_features]
        flat = tf.contrib.layers.flatten(inputs)
        return flat


# concatenate outputs from different layers
def concat(input, name="concat"):
    # expected input is a set of output tensors from different layers
    with tf.name_scope(name):    
        cat = tf.concat(input, axis=0, name=name)
        return cat


# #### Modified LeNet (new)

# In[97]:

def LeNet_2(x):
    
    with tf.name_scope("conv_1"):
        # Input = 32x32x3. Output = 32x32x6.
        conv_1 = create_conv(x, n_kernels=6, kernel_size=(5, 5), strides=[1, 1, 1, 1])
        # Input = 32x32x6. Output = 16x16x6.
        pool_1 = max_pool(conv_1, kernel_size=(2,2), strides=[1,2,2,1])

    with tf.name_scope("conv_2"):
        # Input = 16x16x6. Output = 16x16x16.
        conv_2 = create_conv(pool_1, n_kernels=16, kernel_size=(5, 5), strides=[1, 1, 1, 1])
        # Input = 16x16x16. Output = 8x8x16.
        pool_2 = max_pool(conv_2, kernel_size=(2,2), strides=[1,2,2,1])    
     
    # Flat. Input = 8x8x16. Output = 1024.
    flat_1 = flatten(pool_2)  # = [batch_size, k_features] = [128, 1024]
    
    # Fully Connected. Input = 1024. Output = 120.
    with tf.name_scope("fc_1"):
        fc_1 = create_fc(flat_1, n_inputs=1024, n_outputs=120)
        fc_1 = relu(fc_1)
    
    # Fully Connected. Input = 120. Output = 84.
    with tf.name_scope("fc_2"):
        fc_2 = create_fc(fc_1, n_inputs=120, n_outputs=84)
        fc_2 = relu(fc_2)    
    
    # Fully Connected. Input = 84. Output = 43.
    with tf.name_scope("fc_3"):
        logits = create_fc(fc_2, n_inputs=84, n_outputs=43)
    
    return logits


# ### Modified LeNet (old)

# In[39]:


def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0  # normalized mean
    sigma = 0.1  # normalized stdev

    # Conv. Input = 32x32x3. Output = 32x32x6.
    with tf.name_scope("conv1"):
        conv1_w = tf.Variable(tf.truncated_normal([5, 5, 3, 6], mean=mu, stddev=sigma), name="W")
        conv1_b = tf.Variable(tf.constant(0.01, shape=[6]), name="B")
        conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding="SAME")
        conv1_act = tf.nn.relu(conv1 + conv1_b)
        tf.summary.histogram("weights", conv1_w)
        tf.summary.histogram("biases", conv1_b)
        tf.summary.histogram("activations", conv1_act)
        # Pooling. Input = 32x32x6. Output = 16x16x6.
        conv1 = tf.nn.max_pool(conv1_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Conv. Input = 16x16x6. Output = 16x16x16.
    with tf.name_scope("conv2"):
        conv2_w = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma), name="W")
        conv2_b = tf.Variable(tf.constant(0.01, shape=[16]), name="B")
        conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding="SAME")
        conv2_act = tf.nn.relu(conv2 + conv2_b)
        tf.summary.histogram("weights", conv2_w)
        tf.summary.histogram("biases", conv2_b)
        tf.summary.histogram("activations", conv2_act)
        # Pooling. Input = 16x16x16. Output = 8x8x16.
        conv2 = tf.nn.max_pool(conv2_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    # Flat. Input = 8x8x16. Output = 1024.
    # flat_1 = tf.reshape(conv2, [-1, 8 * 8 * 16])
    flat_1 = flatten(conv2) # = [batch_size, k_features]

    # Fully Connected. Input = 1024. Output = 120.
    with tf.name_scope("fc1"):
        fc1_w = tf.Variable(tf.truncated_normal([1024, 120], mean=mu, stddev=sigma), name="W")
        fc1_b = tf.Variable(tf.constant(0.01, shape=[120]), name="B")
        fc1_act = tf.matmul(flat_1, fc1_w) + fc1_b
        tf.summary.histogram("weights", fc1_w)
        tf.summary.histogram("biases", fc1_b)
        tf.summary.histogram("activations", fc1_act)
        fc1 = tf.nn.relu(fc1_act)
        tf.summary.histogram("fc1/relu", fc1)

    # Fully Connected. Input = 120. Output = 84.
    with tf.name_scope("fc2"):
        fc2_w = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma), name="W")
        fc2_b = tf.Variable(tf.constant(0.01, shape=[84]), name="B")
        fc2_act = tf.matmul(fc1, fc2_w) + fc2_b
        tf.summary.histogram("weights", fc2_w)
        tf.summary.histogram("biases", fc2_b)
        tf.summary.histogram("activations", fc2_act)
        fc2 = tf.nn.relu(fc2_act)
        tf.summary.histogram("fc2/relu", fc2)

    # Fully Connected. Input = 84. Output = 43.
    with tf.name_scope("fc3"):
        fc3_w = tf.Variable(tf.truncated_normal([84, 43], mean=mu, stddev=sigma), name="W")
        fc3_b = tf.Variable(tf.constant(0.01, shape=[43]), name="B")
        logits = tf.matmul(fc2, fc3_w) + fc3_b
        tf.summary.histogram("weights", fc3_w)
        tf.summary.histogram("biases", fc3_b)
        tf.summary.histogram("activations", logits)

    return logits


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[ ]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


# In[101]:

tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)


# In[102]:

logits = LeNet_2(x)

# cross entropy
with tf.name_scope("xent"):
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits), name="xent")
tf.summary.scalar("xent", xent)

# training
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(xent)

# accuracy
with tf.name_scope("accuracy"):
    pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))
tf.summary.scalar("accuracy", acc)


summ = tf.summary.merge_all()

saver = tf.train.Saver()


# In[103]:

def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(acc, feed_dict={x: batch_x, y: batch_y})
#         summ = tf.summary.merge_all()
        total_accuracy += (accuracy * len(batch_x))
    final_accuracy = total_accuracy / num_examples
    return final_accuracy


# In[104]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR) # add 'hparam' for multinetwork tests
    writer.add_graph(sess.graph)    
    num_examples = len(x_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(x_valid, y_valid)
#         print("summ: ", summ)
#         writer.add_summary(summ, i)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
### TensorBoard code ###
# sess.run(tf.global_variables_initializer())
# writer = tf.summary.FileWriter(LOGDIR + hparam)
# writer.add_graph(sess.graph)

# for i in range(2001):
#     batch = mnist.train.next_batch(100)
#     if i % 5 == 0:
#         [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
#         writer.add_summary(s, i)
#     if i % 500 == 0:
#         sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
#         saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
#     sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[ ]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Predict the Sign Type for Each Image

# In[ ]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.


# ### Analyze Performance

# In[ ]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[ ]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

