---
##### Udacity Self-Driving Car Nanodegree
# **Project 2: Traffic Sign Recognition** 
#
---
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

[//]: # (Image References)
[image1]: ./examples/visualization.jpg "Visualization"

##### Rubric Points
In the write-up below, I consider the project's [rubric points](https://review.udacity.com/#!/rubrics/481/view) and describe how I addressed each point wihin my implementation.  
# 
#
## Project Summary
# 
---
# 
### Data Set Summary & Exploration

Throughout this section, I use the Numpy, Pandas, and Matplotlib libraries to explore and visualize the traffic signs data set. 

##### Data Size & Shape
* Size of training set: 
* Size of the validation set:
* Size of test set:
* Shape of a traffic sign image:
* Number of unique classes/labels: 43

![alt text][image1]

##### Data Visualization
Before designing the neural network, I felt it was important to visualize the data in varoius ways to gain some intuition for what the model will "see." This not only informs the model structure and parameters, but it also helps me determine what types of preprocessing operations should be applied to the data (if any). 

There are a few fundamental ways I used visualizations to inform my decisions:
1. **Preview a sample of images (duh!)**
-- Do the images correspond with the expected number of color channels? -- i.e., if channels=3 then images should be color/RGB not grayscale.
-- How clear are the images? Is there anything that makes the signs hard to recognize (e.g. bad weather, darkness, glare, occlusions)?
2. **Look at a sample of the labels**
-- Do the labels make sense? Do they accurately correspond with images in the data set?
3. **Histogram showing the distribution of classes/labels**
-- How balanced is the dataset? Are there certain classes that dominate the dataset? Are there others that are under represented? 

###### Sample of Images & Labels
Here is a sample of original images (one per class) before they undergo any preprocessing. Overall, the image quality is good and the labels make intuitive sense. However, immediately you can notice a few things we'll want to adjust during preprocessing:
* Many of the signs are hard to recognize because the image is dark with low contrast.
* There is little variation in the sign shape and viewing angle. Most of the pictures are taken with straight on view of the sign, which is good for the core data set. However, in real life, signs are viewed from different angles. 
* The signs are void of any visual damage or occlusions. Again, this is good because we need a clean set of training samples, but in real life, signs are sometimes damaged, vandalized, or only partially visible. 

![alt text][image1]
/images/writeup/original-signs.png

# 
# 
### Data Preprocessing
Given the issues identified above, I decided to explore the following preprocessing operations (in addition to the standard practice of _normalization_):
* **Normalization** (standard)
* **Contrast enhancement**
  * I used this Scikit [histogram equalization function](http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist) to enhance local contrast details of the image in regions that are darker or lighter than most of the image. You can see that this not only boosts the contrast, but inherently increases the overall brightness of the image.  
  * ![alt text][image1]: orig_vs_norm.jpg
  * [link to source code]()
* **Augmentation**
  * **Increase total number of images**, so that the model has more training examples to learn from. 
  * **Create an equal distribution of images** (i.e., same number of images per class) so that the model has a sufficient number of training examples in each class. I initially tested models on sets of 3k and 5k images per class, and found that models performed better with more images. I ultimately generated a set of 6k images per class for the final model. 
  * **Apply affine transformations**. Used to generate images with various sets of perturbations. Specifically: rotation, shift, shearing, and zoom. But, I decided _not_ to apply horizontal/vertical flipping as this didn't seem pertinent to real-life use cases. 
  * **Apply ZCA whitening** to accentuate edges.
  * **Apply color transformations**
    * _Color channel shifts_ -- This was done to create slight color derivations, to prevent the model from overfitting specific color shades. This intuitively seemed like a better strategy than grayscaling. 
    * _Grayscaling_ -- This was performed separately _after_ all of the above transformations. Due to the high darkness and low contrast issues, applying grayscale before the other transformations didn't make sense. It would only make them worse. I decided to test the grayscale versions as a separate data set to see if it boosted performance (spoiler aleart: it didn't).

[link to source code]()
![alt text][image1]

##### Examples
Here is a sample of a traffic sign images after the complete set of **normalization, contrast enhancement, and augmentation** listed above.

![alt text][image1]

Here is a sample of images with **grayscaling** applied 

![alt text][image1]

#
### Model Architecture
_Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model._

I tested a variety of models (more than 25 different variations). Ultimately, I settled on a simple and efficient architecture that didn't take forever to train and still delivered great performance. My final model consisted of the following layers:

![architecture diagram][/images/writeup/architecture-diagram.png]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

###### Final model results:
* training set accuracy of **100%**
* validation set accuracy of **99.4%**
* test set accuracy of **98.2%**

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
 

### Test a Model on New Images

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

