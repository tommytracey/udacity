**Udacity Nanodegree &nbsp; | &nbsp; Self-Driving Car Engineer**
# Project 3: Behavioral Cloning

### Goals
The goals of this project are to:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

&nbsp;
### Summary of Results
For this project, I was able to build models that navigated both Track 1 and Track 2 (the challenge track) autonomously. The model for Track 1 was able to navigate the track at the maximum allowed speed of 30 MPH. The model for Track 2 &mdash; which was trained using the weights from the Track 1 model &mdash; was set at 21 MPH, but reached speeds up to 30 MPH in some sections.

Here are the videos that show these models successfully completing one lap around each track.

**Track 1**

<iframe width="560" height="315" src="https://www.youtube.com/embed/rJieV8ADRq4?rel=0" frameborder="0" allowfullscreen></iframe>

##### &nbsp;
**Track 2**

<iframe width="560" height="315" src="https://www.youtube.com/embed/yUl-1NCk2ac?rel=0" frameborder="0" allowfullscreen></iframe>

#### &nbsp;
### Summary of My Approach
You can find a detailed walk-through of my approach and the various parts of my pipeline in the following Jupyter notebooks for [Track 1](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/behavioral-cloning-track1-final.ipynb) and [Track 2](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/behavioral-cloning-track2-final.ipynb). In the next section, I will outline how I addressed the required aspects of this project. But first, I want to highlight a few of the explorations that I conducted, which go beyond the required scope of this project. There are three in particular that I found particularly useful while iterating on my model.

&nbsp;

**1 | &nbsp; Visualizing the Generator Output**

In this project, Udacity recommended that we create a generator to feed data into the model during training. The big advantage of a generator is that you can use a relatively small amount of data to produce a much larger training set.

Another benefit of a generator is that it's efficient. You can augment your data on-the-fly, i.e. in small batches during training, rather than pre-processing a huge dataset beforehand. Generators also allow you to make your training set more dynamic. They can be setup to produce slightly different transformations every time you train the model, which adds variety to your training set and helps reduce overfitting.

However, one of the challenges is that you have less visibility into the training data. You don't have a fixed training set that you can thoroughly investigate to make sure it only has "good data" for training your model. There's a chance your generator could be pumping "bad" (or useless) data into your model. The chances are higher if (like me) you're using a variety of data transformation techniques, which produce thousands of different parameter combinations. I quickly learned that: (a) I need to be careful about how I setup my generator, and (b) I need a way to preview sample outputs from the generator, so I can easily detect problems and make adjustments.

Here is an example. Below you can see a set of training images with their corresponding augmented versions which were output by the generator. The transformations that were applied include: cropping, flipping, smoothing, horizontal and vertical shifts, rotation shifts, brightness shifts, color channel shifts, and resizing. This allowed me to see what the model sees!

[(source code)](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/model.py#L1359)

![generator-output](/results/generator-output.png)

&nbsp;

**2 | &nbsp; Distribution Flattening**

When you capture normal driving data via the simulator, the percentage of that data involves turning the vehicle is quite low. Most of the time you're driving straight. So, to ensure the model sees more turns, I found it useful to flatten the distribution of steering angles being input into the model.

Finding the right distribution took some experimentation. If it's too steep then your car won't learn how to turn effectively. But, if it's too flat, your car won't learn to drive straight and will swerve endlessly around the road.

[(source code)]()


![distribution](/results/distribution-flattened.png)


&nbsp;

**3 | &nbsp; Visualizing Model Filters**

The visualization approach I took superimposes the model filters onto the original training images. To me this is the most intuitive way to understand how your model "sees" the road and which characteristics of the track are factoring into the model's steering predictions. In this way, it's a great tool for identifying weaknesses in your model. This insight helps inform and test further adjustments to the model and/or augmentation of the training data. I only wish I'd done this exploration sooner in the project!

For example, here we see a sample of model filters for right turns (steering angle > 0.35). Notice that in some cases the filters have learned to ignore parts of the road. However, in other cases, the model is ignoring (or at least not focusing on) seemingly important features of the road.

[(source code)](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/model.py#L1529)

![left_turn_filter](/results/filters-left-turns.png)


&nbsp;

**4 | &nbsp; Transfer Learning**

After many attempts, I finally got a solid working model that could consistently navigate around Track 1. Then it was time to focus on Track 2, which is much more challenging. Track 2 has lots more turns and overall the turns are much sharper. Also, the track has lots of hills which shift the horizon and make the track difficult to see.

Given these stark differences, my Track 1 model was not able to generalize to Track 2 &mdash; i.e., I wasn't able to simply run the Track 1 model on Track 2 without crashing the car. However, I was able to repurpose the weights from the Track 1 model to boost the model training process for Track 2.

The results were quite incredible. When I trained the Track 2 model using the Track 1 weights (and Track 2 data), it yielded MUCH better driving behavior than training with Track 2 data from scratch.

[Here's a link](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/model-track2.py#L1416) to the source code which shows how I imported the weights. It's very simple to do in Keras, once you figure out the right approach. You have to recreate the model and use the `load_weights` method just before you compile.

```python
# Weights from Track 1 model
model.load_weights('models/track1.h5')

# Compile and preview the model
model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=['accuracy'])
```

NOTE: There's supposed to be alternative way to do this in Keras, which I wasted 4-5 hours debugging. You should be able to replace the dense layers from your existing model and repurpose just the convolutional layers. Unfortunately, there appears to be a [bug in Keras](https://github.com/fchollet/keras/issues/4802#issuecomment-269323462) that creates a shape mismatch when using the [`load_model` method](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) in this way. However, this approach _does_ work if you use one of the pre-trained CNNs that have their own distinct methods in Keras (e.g. [VGG16](https://keras.io/applications/#vgg16)).

### &nbsp;

---
## Rubric Points
In this section, I consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

&nbsp;
#### Files Submitted & Code Quality


##### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py]() containing the script to create and train the model
* [drive.py]() for driving the car in autonomous mode
* [model.h5]() containing a trained convolution neural network
* writeup_report (this page) summarizing the results

&nbsp;
##### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
$ python drive.py model.h5
```

&nbsp;
##### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The model.py file was exported from [this Jupyter notebook](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/behavioral-cloning-track1-final.ipynb), which outlines the pipeline I used for training and validating the model and contains detailed comments explaining how the code works.

# &nbsp;
### Model Architecture and Training Strategy

##### 1. An appropriate model architecture has been employed

My CNN model consists of:

- a normalization layer
- 5 convolutional layers
- a combination of 3x3 and 5x5 filter sizes with depths between 24 and 64
- a flat layer
- 4 fully connected layers
- ELU activations to introduce nonlinearity throughout the convolutional and connected layers

[(source code)](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/model-track2.py#L1392)

```python
## Create the model ** Track 2 **
# based on NVIDIA model: https://github.com/hdmetor/Nvidia-SelfDriving

model = Sequential()

model.add(Lambda(lambda x: x/255 - 0.5, input_shape=resized_shape))

model.add(Conv2D(24, 5, strides=d_str, padding=d_pad, activation=d_act, kernel_regularizer=reg, name='block1_conv1'))
model.add(Conv2D(36, 5, strides=d_str, padding=d_pad, activation=d_act, kernel_regularizer=reg, name='block1_conv2'))
model.add(Conv2D(48, 3, strides=d_str, padding=d_pad, activation=d_act, kernel_regularizer=reg, name='block1_conv3'))

model.add(Conv2D(64, 3, strides=d_str, padding=d_pad, activation=d_act, kernel_regularizer=reg, name='block2_conv1'))
model.add(Conv2D(64, 3, strides=d_str, padding=d_pad, activation=d_act, kernel_regularizer=reg, name='block2_conv2'))

model.add(Flatten())
model.add(Dense(150, activation=d_act,  kernel_regularizer=reg))
model.add(Dense(50, activation=d_act,  kernel_regularizer=reg))
model.add(Dense(10, activation=d_act,  kernel_regularizer=reg))
model.add(Dense(1))

# Weights from Track 1 model
model.load_weights('models/track1.h5')

# Compile and preview the model
model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=['accuracy'])

model.summary()
```

&nbsp;
##### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. Here are some of the training and validation accuracies recorded for the Track 1 model.

![epochs](/results/track1-training-epochs.png)


I also used L2 regularization to reduce the magnitude of the weights. After some experimentation, I settled on an L2 decay rate of 0.001.

Initially, I experimented with dropouts on the fully connected layers, but I found that they were causing the model to under fit, so I [commented them out](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/model.py#L1473) and used L2 instead.

The ultimate test was running the model through the simulator. If the model was overfitting, it wouldn't be able to stay on the track. For example, I had one model that I'd overfit so badly on turning data that it perpetually drove in circles!

<iframe width="560" height="315" src="https://www.youtube.com/embed/nVyhEbB7k64?rel=0" frameborder="0" allowfullscreen></iframe>

&nbsp;
##### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([source code](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/model.py#L1480)).

&nbsp;
##### 4. Appropriate training data
I recorded a variety of data samples to ensure the model could learn "good" driving behaviors without overfitting the data.

  - center driving data (3 laps)
  - center driving data going the other direction (2 laps)
  - turning data: recorded on sections of the track with sharp turns
  - recovery data: showing the car recovering to the center of the track from the left and right sides of the road
  - trouble spots: recorded on unique parts of the track that the initial models struggled with, e.g. the bridge

# &nbsp;
### Model Architecture and Training Strategy

&nbsp;
##### 1. Solution Design Approach

The overall strategy for finding the right model architecture was to start small and scale up as needed. I tried to avoid having a bloated model with too many parameters and therefore took a long time to train. My working hypothesis was that having a good set of training data was the most important factor to building a model that can autonomously navigate both tracks. So, I wanted a model that allowed me to quickly iterate and test the changes I was making to the dataset. That said, I did experiment with larger models throughout the process.

First, I tried to repurpose the CNN I'd built for the last project (traffic sign classification). But, even after I simplified this model, it still had millions of parameters and took a long time to train (without producing great results). So, I ditched this approach in favor of a well-known NVIDIA model which had a track record (pun intended) solving behavioral cloning for autonomous driving. Again, the overall goal was to get a lightweight model working so I could iterate on the data set. Once I felt the dataset was adequate for training purposes, I invested more time tweaking the model and exploring alternative architectures like [Keras' pre-trained version of VGG]() and [Comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py). But, ultimately I settled on the NVIDIA model given its good performance and relatively small footprint.

To test the model, I first trained it on the data set provided by Udacity. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting the training data. And this was evident by the inability of the car to stay on the road in the simulator. But, at this point I knew the model was capable of learning, so I shifted my focus to the data to address the overfitting problem. Generating a larger and more balanced data set (i.e., one with a higher ratio of turning data) did reduce overfitting. From there, I focused on tuning the regularization parameters.


&nbsp;
##### 2. Final Model Architecture

Here is a visualization of the architecture.

< insert image >

![alt text][image1]

&nbsp;
##### 3. Creation of the Training Set & Training Process

[This notebook](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/behavioral-cloning-track1-final.ipynb) provides a detailed walk-through of all the steps I took to augment and refine the training data for Track 1 (and [this notebook](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/3-behavioral-cloning/behavioral-cloning-track2-final.ipynb) for Track 2). The challenge was to create training data that accurately depicted the driving behaviors the car needed to learn, while also providing enough variation in the data so that the model could generalize its learnings. It was a rigorous process of trial and error, experimenting with different data sources and augmentation techniques, then fine-tuning the model architecture and parameters.

Here are the final training and validation numbers.

- **Track 1 model**:
 - 40,774 training samples (prior to augmentation)
   - 326,192 training samples after augmentation
 - 6,846 validation samples (14%)
 - 3 epochs
 - training loss: 0.0346, validation loss: 0.0241
- **Track 2 model** (repurposing the weights from Track 1 model):
 - 18,502 training samples (prior to augmentation)
   - 74,008 training samples after augmentation
 - 1,194 validation samples (6%)
 - 10 epochs
 - training loss: 0.0635, validation loss = 0.0472
