
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project 3: Behavioral Cloning
# August 2017
# 

# ---

# ## Goal
# The goal of this project is to build a machine learning model that can successfully steer a car around a race track that it's never encountered before.
# 
# The details for this project are located here at [Udacity's Github repo](https://github.com/udacity/CarND-Behavioral-Cloning-P3). My implementation of the project can be found [here at my Github repo](https://github.com/tommytracey/udacity/tree/master/self-driving-nano/projects/3-behavioral-cloning).
# 

# ---
# ## Initial Setup

# #### Import Modules

# In[1]:


import csv
import cv2
import keras
import keras.backend
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf


# #### Verify that Keras is using Tensforflow backend

# The Keras backend uses Theano by default and changing it to Tensorflow can be tricky via Jupyter. Simply updating the config json file `$HOME/.keras/keras.json` as directed in [Keras backend documentation](https://keras.io/backend/) did not work for me. Trying to set it _before or after loading the notebook_ did not work eithter when using:
# 
# `$ os.environ["KERAS_BACKEND"]="tensorflow"`
# 
# The only way I could reliably set Tensorflow as the backend was to use the following command **UPON loading the notebook**:
# 
# `$ KERAS_BACKEND=tensorflow jupyter notebook`
# 
# (NOTE: You can also append `--NotebookApp.iopub_data_rate_limit=10000000000` to the above command if your notebook includes a lot of visualizations. This will help prevent the kernel from crashing and/or causing you to lose your connection to your AWS EC2 instance.)
# 
# Re: the backend, **the cell below only provides a sanity check that the backend is configured as expected**. Note that the version of Tensorflow being used by Keras may be different than the one you typically run in your environment. 
# 
# [This post](https://www.nodalpoint.com/switch-keras-backend/) by Christos-Iraklis Tsatsoulis provides even more detail if you'd like to further understand the issues and automate the setup process.

# In[2]:


print('Keras version: ', keras.__version__)
print('Tensorflow version: ', tf.__version__)
print('Keras backend: ', keras.backend.backend())
print('keras.backend.image_dim_ordering = ', keras.backend.image_dim_ordering())

os.environ["KERAS_BACKEND"] = "tensorflow"
if keras.backend.backend() != 'tensorflow':
    raise BaseException("This script uses other backend")
else:
    keras.backend.set_image_dim_ordering('tf')
    print("\nBackend OK")


# ---
# # Step 1: Load and preview the data

# ---

# ### Data Source A: Provided by Udacity

# In[7]:


## Load UDACITY data from .csv and preview it in Pandas dataframe

udacity_df = pd.read_csv('data/udacity/driving_log.csv', header=0)

print('total rows: ', len(udacity_df))
udacity_df.head()


# ### Data Source B: Self-generated using simulator

# #### Preview the data

# In[3]:


## Load SELF data from .csv and preview it in Pandas dataframe

with open('data/self/driving_log.csv', newline='') as f:
    self_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

columns = self_data[0]
self_df = pd.read_csv('data/self/driving_log.csv', skiprows=[0], names=columns)

print('total rows: ', len(self_df))
self_df.head()


# #### Data Set: Initial Observations
# We can see from the table above that the driving data includes:
# - relative paths to .jpg images from three different camera angles (center, left, right)
# - floating point measurements of the vehicle's steering angle, throttle, brake, and speed
# - the data appears to be time series, although no time stamps are included

# ## Preview the driving images
# 
# The driving images are the training features for our model. We need to look at samples of these images and start thinking about how various characteristics might affect the model (positively or negatively). 

# In[51]:


## Preview a random set of images from each camera angle

# Display visualizations in the notebook
get_ipython().magic('matplotlib inline')

index = random.randint(0, len(udacity_data))
img_dir = 'data/udacity/'

center_img_orig = mpimg.imread(img_dir + udacity_data[index][0])
left_img_orig = mpimg.imread(img_dir + udacity_data[index][1])
right_img_orig = mpimg.imread(img_dir + udacity_data[index][2])

center_steer = udacity_data[index][3]

plt.figure(figsize=(20,5))

plt.subplot2grid((1, 3), (0, 0));
plt.axis('off')
plt.title('left camera')
plt.text(0, left_img_orig.shape[0]+15, ('shape: ' + str(left_img_orig.shape)))
plt.imshow(left_img_orig, cmap="gray")

plt.subplot2grid((1, 3), (0, 1));
plt.axis('off')
plt.title('center camera')
plt.text(0, center_img_orig.shape[0]+15, ('shape: ' + str(center_img_orig.shape)))
plt.text(0, center_img_orig.shape[0]+30, ('steering angle: ' + center_steer))
plt.imshow(center_img_orig, cmap="gray")

plt.subplot2grid((1, 3), (0, 2));
plt.axis('off')
plt.title('right camera')
plt.text(0, right_img_orig.shape[0]+15, ('shape: ' + str(right_img_orig.shape)))
plt.imshow(right_img_orig, cmap="gray")


# #### Driving Images: Initial Observations
# We can see from the images above that:
# - the images are taken in the front of the car (no side or rear angles)
# - each image is 160x320 with 3 RGB color channels
# - there is quite a bit of superfluous data, i.e. data that won't benefit the model; for example the sky, hills, trees in the background, as well as the hood of the car).
# 
# If you view enough images or actually run the simulator, you also see that:
# - there are a lot of turns in the road (duh!), but since the track ultimately ends where it started, there seems to be more turns in one direction than the other
# - the lane markings change shape and color at different points in the track, and at some points there are no markings at all!
# - all of the images are consistently bright; no glare, no darkness, and no shadows that you'd usually encounter with normal driving
# 
# Given the simulation takes place on a race track (not a highway), the road is free of additional cars, traffic signs, lanes, etc. We won't account for these in this project, but a more robust driving model would need training data that included these conditions. 
# 
# That said, many of the other items above can create biases in our model and cause it to overfit the specific driving conditions within this particular simulation. We need to correct for these so that our model learns to drive in a variety of conditions we might find on other tracks. We'll do this by pre-processing and augmenting our training data throughout the sections to follow. But first let's look at our target data (steering angles) to see if there's anything else we need to correct for. 

# ## Examine the steering angles
# The steering angles are our target data for training the model. That is, based on the images fed into the model while the car is driving along the track, the model will predict the appropriate steering angle to navigate the patch of road ahead.

# #### Distribution

# In[22]:


get_ipython().magic('matplotlib inline')

angles = udacity_df.steering.tolist()
num_bins = 35
avg_per_bin = len(angles) / num_bins

hist, bins = np.histogram(angles, num_bins)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

plt.title('Distribution of Steering Angles')
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_per_bin, avg_per_bin), 'k-')
plt.show()


# **Steering Angle Observations**:
# 
# We can see from the graph above that an overwhelming amount of the target data are steering angles close to zero (i.e., when the car is driving straight). This biases our model to drive the car straight and make it's difficult to learn how to navigate turns. 
# 
# It also seems there may be an imbalance in left vs. right turning data (although not as much as I had expected). It's not clear how big of an impact this would have on the model, but there's a chance the model could learn to turn more effectively in one direction than the other. Just to be safe we'll correct for this by ensuring there are equal quantities of left and right steering data after pre-processing. 

# ---
# # Step 2: Data pre-processing
# ---

# ### 2.1 | Overview of Data Sources and Splits
# 
# **Data Sources**: 
# - There are two tracks, but all of the original training and validation data is generated by driving the simulator on **Track 1**.
# - Udacity provides an intial set of "good" data to get us started.
# - Additional data is gathered by running the simulator myself. 
# 
# **Training Data**:
# - The training data set includes the original image data captured from the simulator on Track 1, plus any additional data generated via pre-processing and augmentation. 
# 
# **Validation Data**:
# - The validation data will contain the original image data captured from the simulator on Track 1, with only a pre-process to create a more balanced distribution (i.e., reduce the 'drive straight' bias). No other pre-processing or augmentation is included. This ensures we can properly validate whether changes we're making to the model or training data are increasing or decreasing the model's performance. 
# 
# **Test Data**:
# - We'll test the model's ability to generalize by running it on **Track 2**. So, the simulator input images from Track 2 will serve as our test data.

# ### 2.2 | Data Hygiene
# 
# Before going any further there are few aspects of the data we need to cleanup to make the data easier to work with.
# 
# 1. We're only using the steering data to train the model, so we can prune out the other measurements (throttle, brake, and speed).
# 2. Remove the directory from image path data. We'll be moving our data around and we only want the image filename. 
# 3. Cast all of the steering data as floats. In the .csv they're cast as strings. 

# #### Udacity data hygiene

# In[14]:


# Load UDACITY data into a list
with open('data/udacity/driving_log.csv', newline='') as f:
    udacity_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
# column references: 0=center_img, 1=left_img, 2=right_img, 3=steering, 4=throttle, 5=brake, 6=speed

udacity_clean = []

for row in udacity_data[1:]:
    # Remove directory from image paths
    center = row[0].split('/')[-1]
    left = row[1].split('/')[-1]
    right = row[2].split('/')[-1]
    
    # Only grab the steering data and cast as float
    angle = float(row[3])
    
    udacity_clean.append([center, left, right, angle])

print('number of records: ', len(udacity_clean))
print('\nfirst 3 records:\n', udacity_clean[0:3])


# In[83]:


udacity_clean[0:150]


# #### Self-generated data hygiene

# In[15]:


# Load SELF data into a list
with open('data/self/driving_log.csv', newline='') as f:
    self_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
# column references: 0=center_img, 1=left_img, 2=right_img, 3=steering, 4=throttle, 5=brake, 6=speed

self_clean = []

for row in self_data[1:]:
    # Remove directory from image paths
    center = row[0].split('/')[-1]
    left = row[1].split('/')[-1]
    right = row[2].split('/')[-1]
    
    # Only grab the steering data and cast as float
    angle = float(row[3])
    
    self_clean.append([center, left, right, angle])

print('number of records: ', len(self_clean))
print('\nfirst 3 records:\n', self_clean[0:3])


# ### 2.3 | Seperate C/L/R camera data and adjust steering angles for left and right turns
# Right now, each row of the data set contains three camera angles (center, left, right) and one steering angle which pertains to the center camera. In order to utilize all of the different camera data, we need to: 
# 
# 1. Separate the data for each of the different camera angles (only one camera angle per row)
# 2. Adjust the steering angles for the left and right cameras _while the car is turning_. This will compensate for their respective vantage points relative to the center of the car. That is, the steering angle for a right turn should be sharper from the persective of the left camera (and vice versa). 

# In[84]:


turn_thresh = 0.08   # the angle threshold used to identify left and right turns

def steer_correct(angle):
    '''Calculates the steering angle factor applied to left and right cameras while turning.
    
    Ensures that sharper turns receive a low multiplication factor than softer turns. 
    '''
    corr_factor = (-1.15 * abs(angle)) + 2.15
    new_angle = corr_factor * angle
    return new_angle  


# In[88]:


udacity_3cam = []

for row in udacity_clean:
    # left camera with adjusted steering angle
    if row[3] > turn_thresh:    # if right turn
        img_left = row[1]
        ang_left = steer_correct(row[3])
        udacity_3cam.append([img_left, ang_left])
    
    # right camera with adjusted steering angle
    if row[3] < -turn_thresh:    # if left turn
        img_right = row[2]
        ang_right = steer_correct(row[3])
        udacity_3cam.append([img_right, ang_right])
    
    # center camera and steering angle are kept as is
    img_center, ang_center = row[0], row[3]
    udacity_3cam.append([img_center, ang_center])


# ### 2.4 | Crop & Flip

# In[37]:


## Define cropping dimensions

# Crop settings provided to Keras Cropping2D layer
crop_set = (60, 20), (20, 20)   # number of pixels to remove from (top, bottom), (left, right)
    
# Define image shapes 
orig_shape = center_img_orig.shape  # (160, 320, 3)
crop_shape = (
    orig_shape[0]-(crop_set[0][0]+crop_set[0][1]), \
    orig_shape[1]-(crop_set[1][0]+crop_set[1][1]),  \
    orig_shape[2]
)

# Define resulting crop points for preview cell below
h1, h2 = (crop_set[0][0], orig_shape[0]-crop_set[0][1])
w1, w2 = (crop_set[1][0], orig_shape[1]-crop_set[1][1])


print('orig_shape: ', orig_shape)
print('crop_shape: ', crop_shape)
print('crop points [h1:h2,w1:w2] = [{}:{},{}:{}]'.format(h1, h2, w1, w2))


# In[113]:


## Preview cropped and flipped images

index = random.randint(0, len(udacity_data))
img_dir = 'data/udacity/'

# Select a random set of images to crop
center_img_crop = mpimg.imread(img_dir + udacity_data[index][0])[h1:h2,w1:w2]
left_img_crop = mpimg.imread(img_dir + udacity_data[index][1])[h1:h2,w1:w2]
right_img_crop = mpimg.imread(img_dir + udacity_data[index][2])[h1:h2,w1:w2]

# Create flipped versions
center_img_flip = cv2.flip(center_img_crop, 1)
left_img_flip = cv2.flip(left_img_crop, 1)
right_img_flip = cv2.flip(right_img_crop, 1)

# Calculate steering angles
center_steer = float(udacity_data[index][3])
left_steer = None
left_steer_flip = None 
right_steer = None
right_steer_flip = None
if center_steer > turn_thresh:
    left_steer = steer_correct(center_steer)
    left_steer_flip = -steer_correct(center_steer)
if center_steer < -turn_thresh:
    right_steer = steer_correct(center_steer)
    right_steer_flip = -steer_correct(center_steer)
    
# Display visualizations in the notebook
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(20,6))

# Cropped versions
plt.subplot2grid((2, 3), (0, 0));
plt.axis('off')
plt.title('Left Camera (cropped)')
plt.text(0, left_img_crop.shape[0]+15, ('shape: ' + str(left_img_crop.shape)))
plt.text(0, left_img_crop.shape[0]+30, ('steering angle: ' + str(left_steer)))
plt.imshow(left_img_crop, cmap="gray")

plt.subplot2grid((2, 3), (0, 1));
plt.axis('off')
plt.title('Center Camera (cropped)')
plt.text(0, center_img_crop.shape[0]+15, ('shape: ' + str(center_img_crop.shape)))
plt.text(0, center_img_crop.shape[0]+30, ('steering angle: ' + str(center_steer)))
plt.imshow(center_img_crop, cmap="gray")

plt.subplot2grid((2, 3), (0, 2));
plt.axis('off')
plt.title('Right Camera (cropped)')
plt.text(0, right_img_crop.shape[0]+15, ('shape: ' + str(right_img_crop.shape)))
plt.text(0, right_img_crop.shape[0]+30, ('steering angle: ' + str(right_steer)))
plt.imshow(right_img_crop, cmap="gray")

# Flipped version
plt.subplot2grid((2, 3), (1, 0));
plt.axis('off')
plt.title('Left Camera (cropped + flipped)')
plt.text(0, left_img_flip.shape[0]+15, ('shape: ' + str(left_img_flip.shape)))
plt.text(0, left_img_flip.shape[0]+30, ('steering angle: ' + str(left_steer_flip)))
plt.imshow(left_img_flip, cmap="gray")

plt.subplot2grid((2, 3), (1, 1));
plt.axis('off')
plt.title('Center Camera (cropped + flipped)')
plt.text(0, center_img_flip.shape[0]+15, ('shape: ' + str(center_img_flip.shape)))
plt.text(0, center_img_flip.shape[0]+30, ('steering angle: ' + str(-center_steer)))
plt.imshow(center_img_flip, cmap="gray")

plt.subplot2grid((2, 3), (1, 2));
plt.axis('off')
plt.title('Right Camera (cropped + flipped)')
plt.text(0, right_img_flip.shape[0]+15, ('shape: ' + str(right_img_flip.shape)))
plt.text(0, right_img_flip.shape[0]+30, ('steering angle: ' + str(right_steer_flip)))
plt.imshow(right_img_flip, cmap="gray")


# ### 2.5 | Other Transformations

# Generates batches of tensor image data that is augmented based on a chosen set of tranformation parameters (e.g. rotation, shift, shear, zoom).

# In[ ]:


## Color transformations
# Brightness
# HSA
# Color channels
# Shadows

## Affine transformations
# Shift
# Shear

# Create equal distribution?


# In[25]:


# ## Apply affine and color transformations 

# # Transformations applied to RBG training images *ORIGINAL from Project 2*
# datagen = keras.preprocessing.image.ImageDataGenerator(
#     zca_whitening=True,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.15,
#     zoom_range=0.15,
#     channel_shift_range=0.1,
#     fill_mode='nearest',
#     horizontal_flip=False,
#     vertical_flip=False
# )


# In[ ]:


# NEW augmenation function

datagen = keras.preprocessing.image.ImageDataGenerator(
    zca_whitening=True,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.15,
    channel_shift_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False
)


# In[ ]:


## Create AUGMENTED training sets

from sklearn.utils import shuffle

max_images_per_class = 6000
X_train_aug = np.empty((0, 32, 32, 3))
y_train_aug = np.empty(0, dtype='uint8')

print('Augmenting Image Data...')
for i in tqdm(range(n_labels)):
    index = [y_train==i]
    i_class_images = X_train_norm[y_train==i]
    i_class_y = y_train[y_train==i]

    i_X_train_aug = np.copy(i_class_images)
    i_y_train_aug = np.copy(i_class_y)

    for X,y in datagen.flow(i_class_images, i_class_y, batch_size=len(i_class_y), seed=113+i*17):            
        i_X_train_aug = np.append(i_X_train_aug, X, axis=0)
        i_y_train_aug = np.append(i_y_train_aug, y, axis=0)

        if len(i_X_train_aug) >= max_images_per_class:
            break

    X_train_aug = np.append(X_train_aug, i_X_train_aug[:max_images_per_class], axis=0)
    y_train_aug = np.append(y_train_aug, i_y_train_aug[:max_images_per_class], axis=0)              

X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=113)

print('Augmentation Complete.')



# In[ ]:


print('new number of training images: ', X_train_aug.shape[0])
print('verify image shape: ', X_train_aug.shape[1:])


# In[ ]:


## Display sample of the AUGMENTED training images, 1 image for each class

fig = plt.figure(figsize=(22,35), tight_layout={'h_pad':5})

for i in range(n_labels):
    image_key = np.where(y_train_aug==i)
    img = X_train_aug[image_key[0][0]]
    ax = fig.add_subplot(int(n_labels/4)+1,4,i+1) 
    ax.imshow(img, interpolation='none')
    plt.title("[%02d] %s" % (i, labels_dict[y_train_aug[image_key[0][0]]]))
    plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:


## Display a broader sample of AUGMENTED images, 12 from each class

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

images_labels = list(train_images_dict.keys())
for image_key in images_labels:        
    subplots = 12
    fig, axes = plt.subplots(1,subplots)
    images_total = len(train_images_dict[image_key])
    print('[{}] {}'.format(image_key, labels_dict[image_key]))
    offset = images_total // subplots
    images = [train_images_dict[image_key][j*offset] for j in range(subplots)]
    for i, (image, ax) in enumerate(zip(images, axes)):
        ax.imshow(image.squeeze())
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

train_images_dict.clear()


# ### 2.6 | Training and Validation sets
# 

# In[117]:


## Create training and validation sets

X_train = [i[0] for i in udacity_3cam]
y_train = [i[1] for i in udacity_3cam]

X_train, X_valid, y_train, y_valid =                 train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    
    
# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_valid = len(X_valid)


print("Number of training examples: ", n_train)
print("Number of validation examples: ", n_valid)
print("----------\nVerify Totals: {} should equal {} ".format((n_train+n_valid), len(udacity_3cam)))


# ### 2.7 | Batch Generator

# In[123]:


def generator(images, angles, source, val=False):
    '''Creates batches of augmented images to feed into the model. 
    
    For each input image, four different versions are generated:
    img_1 : original version
    img_2 : flipped version of 1
    img_3 : version of 1 with other random transformations (for training only)
    img_4 : version of 2 with other random transformations (for training only)
    
    '''
    images, angles = shuffle(images, angles)
    X = []  # images batch
    y = []  # angles batch
    
    while True:
        for i in range(len(angles)):
            # retrieve the image from local directory
            img_path = img_dir(source) + str(images[i])
            img = mpimg.imread(img_path.strip())
            
            # generate original and flipped versions
            img_1, ang_1 = (img, angles[i])
            img_2, ang_2 = (cv2.flip(img_1, 1), -angles[i])
            
            # add to batch
            X.extend([img_1, img_2])
            y.extend([ang_1, ang_2])
            
#             # augmentation process; for training images only
#             if not val:
#                 # apply other transformations
#                 img_3 = datagen.flow()
#                 img_4 = 
                
#                 # add to batch
#                 X.append(img_3, img_4)
#                 y.append(ang_3, ang_4)
            
            if len(y) == batch_size:
                X, y = shuffle(np.array(X), np.array(y))
                yield (X, y)
                X, y = ([],[])
#                 images, angles = shuffle(images, angles)


# ---
# ## Step 3: Model Architecture
# ---

# ### Model

# In[121]:


## Global variables and parameters

MODEL_DIR = "models/"

# Image directory
source = 'udacity'
def img_dir(source):
    return "data/" + source + "/IMG/"

epochs = 5
batch_size = 128

lr = 1e-4          # learning rate
reg = l2(1e-3)     # L2 reg
drop = 0.5

strides = (2, 2)
act = 'relu'
default_pad = 'same'


# In[120]:


model = Sequential()

model.add(Lambda(lambda x: x/255 - 0.5, input_shape=orig_shape))
model.add(Cropping2D(cropping=((60, 20), (20, 20))))

model.add(Convolution2D(24, 5, strides=strides, padding=default_pad, activation=act,  kernel_regularizer=reg))
model.add(Convolution2D(48, 5, strides=strides, padding=default_pad, activation=act,  kernel_regularizer=reg))
model.add(Convolution2D(96, 5, strides=strides, padding=default_pad, activation=act,  kernel_regularizer=reg))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Convolution2D(128, 3, strides=strides, padding=default_pad, activation=act,  kernel_regularizer=reg))
model.add(Convolution2D(128, 3, strides=strides, padding=default_pad, activation=act,  kernel_regularizer=reg))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Flatten())
model.add(Dense(100, kernel_regularizer=reg))
# model.add(Dropout(drop))
model.add(Dense(50, kernel_regularizer=reg))
# model.add(Dropout(drop))
# model.add(Dense(16, kernel_regularizer=reg))
# model.add(Dropout(drop))
model.add(Dense(25, kernel_regularizer=reg))
model.add(Dense(1))

# Compile and preview the model
model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=['accuracy'])

model.summary()


# In[ ]:





# In[124]:


## Train and save the model

n_img_gen = 2  # image output:input ratio from generator

train_steps = (n_img_gen * len(X_train)) // batch_size
val_steps = len(X_valid) // batch_size

# train_samples = train_steps * batch_size
# val_samples = val_steps * batch_size

train_gen = generator(X_train, y_train, source=source, val=False)
val_gen = generator(X_valid, y_valid, source=source, val=True)

model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,                     validation_data=val_gen, validation_steps=val_steps)

print('\nDone Training')

# Save model and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")
print("Saved model to disk")


# In[ ]:




