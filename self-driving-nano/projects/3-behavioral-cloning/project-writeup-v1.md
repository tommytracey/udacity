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

### Summary of Results
For this project, I was able to build models that navigated both Track 1 and Track 2 (the challenge track) autonomously. The model for Track 1 was able to navigate the track at the maximum allowed speed of 30 MPH. The model for Track 2 &mdash; which was trained using the weights from the Track 1 model &mdash; was set at 21 MPH, but reached speeds up to 30 MPH in some sections.

Here are the videos that show these models successfully completing one lap around each track.

[embed video 1]
[embed video 2]

### Summary of My Approach
You can find a detailed walk-through of my approach and the various parts of my pipeline in the following Jupyter notebooks for [Track 1]() and [Track 2](). In the next section, I will outline how I addressed the required aspects of this project. But first, I want to highlight a few of the explorations that I conducted, which go beyond the required scope of this project. There are three in particular that I found particularly useful while iterating on my model.

**1 | Visualizing the Generator Output**

In this project, Udacity recommended that we create a generator to feed data into the model during training. The big advantage of a generator is that you can use a relatively small amount of data to produce a much larger training set.

Another benefit of a generator is that it's efficient. You can augment your data on-the-fly, i.e. in small batches during training, rather than pre-processing a huge dataset beforehand. Generators also allow you to make your training set more dynamic. They can be setup to produce slightly different transformations every time you train the model, which adds variety to your training set and helps reduce overfitting.

However, one of the challenges is that you have less visibility into the training data. You don't have a fixed training set that you can thoroughly investigate to make sure it only has "good data" for training your model. There's a chance your generator could be pumping "bad" (or useless) data into your model. The chances are higher if (like me) you're using a variety of data transformation techniques, which produce thousands of different parameter combinations. I quickly learned that: (a) I need to be careful about how I setup my generator, and (b) I need a way to preview sample outputs from the generator, so I can easily detect problems and make adjustments.

Here is an example. Below you can see a set of training images with their corresponding augmented versions which were output by the generator. The transformations that were applied include: cropping, flipping, smoothing, horizontal and vertical shifts, rotation shifts, brightness shifts, color channel shifts, and resizing. This allowed me to see what the model sees!

[(link to source code)]()

<img> insert image </img>



**2 | Distribution Flattening**


**3 | Visualizing Model Filters**


**4 | Transfer Learning**


&nbsp;

---
## Rubric Points
In this section, I consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


#### Files Submitted & Code Quality

1 . Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results


2 . Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for finding the right model architecture was to start small and scale up as needed. I tried to avoid having a bloated model with too many parameters and therefore took a long time to train. My working hypothesis was that having a good set of training data was the most important factor to building a model that can autonomously navigate both tracks. So, I wanted a model that allowed me to quickly iterate and test the changes I was making to the dataset.

First, I tried to repurpose the CNN I'd built for the last project (traffic sign classification). But, even after I simplified this model, it still had millions of parameters and took a long time to train (without producing great results). So, I ditched this approach in favor of a well-known NVIDIA model which had a track record (pun intended) solving behavioral cloning for autonomous driving. Again, the overall goal was to get a lightweight model working so I could iterate on the data set. Once I felt the dataset was adequate for training purposes, I would invest more time tweaking the model and exploring alternative architectures like Comma.ai.

To test the model, I first trained it on the data set provided by Udacity. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. And this was evident by the inability of the car to stay on the road in the simulator. But, I knew at this point the model was capable of learning, so I shifted my focus to the data. A bigger and more diverse data set (i.e., one with a higher ratio of turning data) should reduce the overfitting problem. Although if overfitting is still a problem, I can then start adding regularization to the model. Essentially, I'd rather have the model overfit than underfit in the beginning because the tuning process is more straightforward (i.e., add dropout, L2 regularization, etc.)

This notebook provides a detailed walk-through of all the steps I took to augment and refine the training data. It was a quite rigorous process. The challenge was to create a version of the Track 1 dataset that accurately depicted the driving behaviors the car needed to learn, while also providing enough variation in the data so that the model could generalize its learnings to driving successfully on Track 2.

Ultimately, I was able to get the car to complete a full lap on Track 1 at 30 MPH and on Track 2 at 15 MPH with only training on Track 1 data. Getting the car to to go faster on Track 2 required the addition of Track 2 training data. Once this was added, I was able to get the car up to 25 MPH on Track 2.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
