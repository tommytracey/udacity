<img src='images/writeup/collage-1b.jpg'>

##### Udacity Self-Driving Car Nanodegree
# **Project 2: Traffic Sign Recognition** 
###

The goal of this project is to build a neural network that recognizes traffic signs in Germany. 

Udacity's benchmark for the project is to achieve at least 93% accuracy (on the validation set). However, my personal goal was to surpass the human level performance benchmark of 98.8% accuracy identified in [this paper](https://arxiv.org/pdf/1511.02992.pdf) by Mrinal Haloi from the Indian Institute of Technology.

The basic steps of the project are as follows:
1. Load the data set (see below for links to the project data set)
1. Explore, summarize and visualize the data set
1. Design, train and test a model architecture
1. Use the model to make predictions on new images
1. Analyze the softmax probabilities of the new images
1. Summarize the results with a written report

##### Project Notebook
My code and a detailed view of the outputs for each step are outlined here in this [Jupyter Notebook](). You can also view just the python code via the notebook's corresponding [.py export file](). 


##### Rubric Points
In the write-up below, I consider the project's [rubric points](https://review.udacity.com/#!/rubrics/481/view) and describe how I addressed each point wihin my implementation.  
### 
---
### 
## Data Summary & Exploration

Throughout this section, I use the Numpy, Pandas, and Matplotlib libraries to explore and visualize the traffic signs data set. 

### Data Size & Shape
I used the default testing splits provided by Udacity.

* Size of training set: **34,799**
* Size of the validation set: **4,410**
* Size of test set: **12,630**
* Shape of a traffic sign image: **(32, 32, 3)**
* Number of unique classes/labels: **43**

[(link to source code)]()

### Data Visualization
Before designing the neural network, I felt it was important to visualize the data in varoius ways to gain some intuition for what the model will "see." This not only informs the model structure and parameters, but it also helps me determine what types of preprocessing operations should be applied to the data (if any). 

There are a few fundamental ways I used visualizations to inform my decisions:
1. **Preview a sample of images (duh!)**
   
   Do the images correspond with the expected number of color channels? -- i.e., if channels=3 then images should be color/RGB not grayscale.
   
   How clear are the images? Is there anything that makes the signs hard to recognize (e.g. bad weather, darkness, glare, occlusions)?
2. **Look at a sample of the labels**
   
   Do the labels make sense? Do they accurately correspond with images in the data set?
3. **Histogram showing the distribution of classes/labels**
   
   How balanced is the dataset? Are there certain classes that dominate the dataset? Are there others that are under represented? 

### Sample of Images & Labels
Here is a sample of original images (one per class) before they undergo any preprocessing. Overall, the image quality is good and the labels make intuitive sense. However, immediately you can notice a few things we'll want to adjust during preprocessing:
* Many of the signs are hard to recognize because the **images are dark and have low contrast**.
* There is **little variation in the sign shape and viewing angle**. Most of the pictures are taken with straight on view of the sign, which is good for the core data set. However, in real life, signs are viewed from different angles. 
* The **signs are void of any deformations or occlusions**. Again, this is good because we need a clean set of training samples, but in real life, signs are sometimes damaged, vandalized, or only partially visible. Essentially, we want the model to recognize signs even when the shape is in someway distorted, much like humans can. So, augmenting the training set with distortions is important. 

<img src='images/writeup/original-signs.jpg' width="100%"/>

### Class/Label Distribution
As you can see, the distribution is not uniform. The largest classes have 10x the number of traffic sign images than the smallest classes. This is expected given that in real-life there are certain signs which appear more often than others. However, when training the model, I wanted a more uniform distribution so that each class has the same number of training examples (and the model therefore has an equal number of opportunities to learn each sign). 

<img src='images/writeup/distribution.png' width="100%"/>


###
---
## Data Preprocessing
Given the issues identified above, I decided to explore the following preprocessing operations (in addition to the standard practice of _normalization_):

* __Normalization__ (standard)
* __Contrast enhancement__ (done as part of normalization process)
  * I used this Scikit [histogram equalization function](http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist), which not only normalizes the images, but also enhances local contrast details in regions that are darker or lighter than most of the image. You can see from the image sample below this also inherently increases the brightness of the image. [(link to source code)]()

   <img src='images/writeup/orig_vs_norm.jpg' width="25%"/>

* __Augmentation__
  * __Increase total number of images__, so that the model has more training examples to learn from. 
  * __Create an equal distribution of images__ (i.e., same number of images per class) so that the model has a sufficient number of training examples in each class. I initially tested models on sets of 3k and 5k images per class, and found that models performed better with more images. I ultimately generated a set of 6k images per class for the final model. 
  * __Apply affine transformations__. Used to generate images with various sets of perturbations. Specifically: rotation, shift, shearing, and zoom. But, I decided _not_ to apply horizontal/vertical flipping as this didn't seem pertinent to real-life use cases. 
  * __Apply ZCA whitening__ to accentuate edges.
  * __Apply color transformations__
    * _Color channel shifts_ -- This was done to create slight color derivations, to prevent the model from overfitting specific color shades. This intuitively seemed like a better strategy than grayscaling. 
    * _Grayscaling_ -- This was performed separately _after_ all of the above transformations. Due to the high darkness and low contrast issues, applying grayscale before the other transformations didn't make sense. It would only make them worse. I decided to test the grayscale versions as a separate data set to see if it boosted performance (spoiler aleart: it didn't).

Here is the snippet of code that takes the already normalized images (with contrast enhanced) and applies the transformations listed above. It outputs a new training set with 6k images per class, including the set of normalized training images. 
[(link to source code)]()

<img src='images/writeup/keras-aug-function.jpg' width="60%"/>

<img src='images/writeup/aug-function.jpg' width="96%"/>

<img src='images/writeup/aug-count.jpg' width="62%"/>


### Augmented Image Samples
Here is a sample of a traffic sign images after the complete set of **normalization, contrast enhancement, and augmentation** listed above.

<img src='images/writeup/augmented-sample.jpg' width="100%"/>


### Grayscaling
Here is a sample of images with **grayscaling** then applied. At first glance, it doesn't appear that grayscaling improves the images in any meaningful way. So, my hypothesis was that the grayscaled versions would perform the same or worse than the augmented images (this turned out to be correct).

<img src='images/writeup/grayscale-sample.jpg' width="85%"/>


---
## Model Architecture

I tested a variety of models (more than 25 different combinations). Ultimately, I settled on a relatively small and simple architecture that was easy to train and still delivered great performance. My final model consisted of the following layers:

<img src='images/writeup/architecture-diagram.png' width="60%"/>


###
Here is a snapshot of the code. You can see that I use: (a) a relu activation on every layer, (b) maxpooling on the alternating convolutional layers with a 5x5 filter, and (c) dropouts on the two fully connected layers with a 0.5 keep probability.

[(link to source code)]()

<img src='images/writeup/final-model-code.jpg' width="100%"/>

###
Here are my **training and loss functions**. You can see that I use the AdamOptimizer to take advantage of its built-in hyperparameter tuning, which varies the learning rate based on moving averages (momentum) to help the model converge faster, without having to manually tune it myself. You'll notice that I also use L2 regularization to help prevent overfitting.

<img src='images/writeup/training-and-loss-functions.jpg' width="100%"/>

###
Here are the **hyperparameters** I used. My goal was to get the model to converge in less than 50 epochs. Essentially, given time constraints, I didn't want to spend more than two hours training the model. Everything else is pretty standard. Although, I did decrease my L2 decay rate (i.e. lower penalty on weights) during the tuning process, which yielded an incremental lift in performance.  

<img src='images/writeup/hyperparams.jpg' width="37%"/>

###
Here is the output when I construct the graph. I use print statements to verify that the model structure matches my expectations. I find this very useful as it's easy to get confused when you're tweaking and testing lots of different models. Especially at 3am.  =)

<img src='images/writeup/final-graph-output.jpg' width="50%"/>

###
### Final model results:
* training set accuracy of **100%**
* validation set accuracy of **99.4%**
* test set accuracy of **98.2%**

###
### Model Iteration & Tuning
I'll try to summarize the approach I took to find a solution that exceeded the benchmark validation set accuracy of 0.93. Although some of the details got lost in the fog of war. I battled with these models for too many days. If you're curious, you can view a fairly complete list of the models I tested [here](data/model-performance-summary-v2.xlsx). 

#### Phase 1
The first steps were to get the most basic version of LeNet's CNN running and begin tuning it. I got 83% accuracy without any modifications to the model or preprocssing of the training data. Adding regularization and tuning the hyperparameters made the performance worse. So, I started to explore different types of architectures.

#### Phase 2
This is where I started making mistakes that cost me a lot of time (although I learned a lot in the process). In hindsight, I should have done two simple things: (1) start applying some basic preprocessing to the data and testing the performance impact, and (2) keep iterating on the LeNet architecture by incrementally adding and deepening the layers. 

Instead, I started explore different architectures such as [DenseNets](https://arxiv.org/abs/1608.06993). 

<img src='images/writeup/densenet.jpg' width="45%"/>

DenseNets didn't seem overly complex at the time, and I probably could have gotten them working if I'd just focused on this. However, in parallel, I tried to get Tensorboard working. Trying to both of these at once was a disaster. In short, creating DenseNets requires a lot of nested functions to create all of the various blocks of convuloutional layers. Getting the Tensorboard namespaces to work, getting all of your variables to initialize properly, and getting all of the data to flow properly in and out of these blocks was a huge challenge. After a ton of research and trial and error, I ultimately abandoned this path. ¯\_(ツ)_/¯

I then tried to implement the (much simpler) inception framework discussed by Vincent during the lectures. After some trial and error, I got an inception network running. But, I couldn't get them to perform better than 80% validation accuracy, so I abandoned this path as well. I believe this approach could have worked, but by this point, I wanted to get back to the basics. So, I decided to focus on data preprocessing and iterating on the basic LeNet architecture (which I should have done from the beginning! Arg.)

#### Phase 3
After a day of sleep, yoga, and meditation to clear my head...I got back to basics. 

I started by applying simple transformations to the data and testing simple adjustments to the LeNet architecture. Model performance started to improve, but I still had a bias problem. In the beginning, my models were consistently overfitting the training data and therefore my training accuracy was high but my validation accuracy was still low. 

This is a summary of the tactics I deployed to improve performance.

| Model			        |     Validation Accuracy	        					| 
|:---------------------|:----------------------------------------------:| 
| Basic LeNet      		                                    | 82.6%   	| 
| LeNet + bias init =0 (instead of 0.01)    			         | 85.2%		|
| LeNet + bias init + contrast enhancement					   | 92.9%		|
| LeNet + bias init + contrast + augmentation v1 	         | 94.9%		|
| LeNet + bias init + contrast + aug_v1 + deeper layers		| 97.5%     |
| LeNet + bias init + contrast + aug_v1 + more layers	+ regularization		| 98.1%     |
| LeNet + bias init + contrast + aug_v2 + more layers	+ reg tuning		| 99.0%   |
| LeNet + bias init + contrast + aug_v2 + more layers	+ reg tuning + grayscale		| 95.8%  |
| LeNet + bias init + contrast + aug_v2 + more layers	+ reg tuning + more training images	+ more epochs	| 99.4%  |

More details regarding the tactics above:

* __bias initialization__ &mdash;
* __contrast enhancement__ &mdash;
* __augmentation v1 vs v2__ &mdash;
* __regularization__ &mdash;
* __grayscale__ &mdash;
* __more training images + more epochs__ &mdash;


---
Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
###
---
## Test a Model on New Images

####1. Choose at least five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

