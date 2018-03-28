### Self-Driving Car Engineer Nanodegree
# Project: Finding Lane Lines on the Road
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

## Goal
The goal of this project is to make a pipeline that finds lane lines on the road.

## Overview
When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

---
# Results
The sections below outline the work I completed as part of this project. The Jupyter Notebook document containing the source code is located [here](https://github.com/tommytracey/udacity/blob/master/self-driving-nano/projects/1-lane-lines/final_submission/P1-v3.ipynb).


### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My final pipeline code is available [here](http://localhost:8888/notebooks/P1-v3.ipynb#Video-Pipeline). The pipeline consists of five main steps:

#### Step 1
First, I convert the images to grayscale. For this initial exercise, this simplifies the process since we're relying mostly on variations in brightness (not color) to identify lane lines.

<img src='results/1-code.png' width="60%"/>

<img src='results/1-grayscale.png' />


#### Step 2
I then apply Gaussian smoothing. This reduces the amount of detail and noise in the image, and in turn, hides the fainter edges that we don't care about. This makes it easier to identify the most prominent edges in the next step.

<img src='results/2-code.png' width="65%"/>

<img src='results/2-gaussian.png' />


#### Step 3
Next, I apply the Canny transform to identify edges within the image. Edges are areas with the strongest gradient -- i.e., areas where there's a large difference in brightness between adjacent pixels. In photography, you might think of these as the areas with the highest contrast. These edges allow us to detect the boundaries of objects within the image. In this case, we're looking for the boundaries of the brighter driving lane lines against the darker background of the highway pavement (but, this process will also detect the edges of other objects). The Canny algorithm allows us to identify the individual pixels where these edges are the strongest (above two given thresholds).

<img src='results/3-code.png' width="60%"/>

<img src='results/3-canny.png' />


#### Step 4
Then, I create a mask to define the part of the image where we expect the lane lines to be (aka the 'region of interest'). Obviously, we should look for lane lines on the road immediately in front of the car; we should not look for them in the sky. But perhaps someday when we have flying cars! 8')

<img src='results/4-code.png' width="95%"/>

<img src='results/4-region-of-interest.png' />


#### Step 5
Finally, I detect the lane lines using the probabilistic Hough transform method. This method takes all of the edge points detected by the Canny algorithm and converts them to lines in Hough space.

Each point along lines in Hough space represents a possible set of line parameters for edge points within the image space. When multiple lines intersect in Hough space, we know we've found a set of line parameters that represents multiple edge points within the image space. The more interesections in Hough space the better, as this means we've found a clearly defined line within the image.

<img src='results/5-hough-diagram.png' width="60%"/>

The Hough algorithm can also be tuned with various parameters. For example, in my model, I expect relevant lines to have at least 60 pixels.

<img src='results/5-hough-params.png' width="80%"/>

The Hough output lines are then averaged to create two distinct lane lines. The final two lane lines are then drawn onto a separate image and applied to the original image as a transparency.

<img src='results/5a-hough-lines.png' />

<img src='results/5b-final-image.png' />


In order convert the various lines identified by the Hough transform into two single lines for the left and right lanes, I made a number of modifications to the original draw_lines() function:

- First, I divide the Hough output lines into two groups: those with positive slope vs. those with negative slope.
- I then average all of the x,y coordinates for each group to derive a single set of line parameters which best describes the group.
- The two sets of line parameters for each individual frame of the video are then logged. From the log, I calculate a moving average based on the most recent set of frames. These moving average line parameters are then used to draw the final lane line guides.


[Here](http://localhost:8888/notebooks/P1-v3.ipynb#Modified-Helper-Functions-for-Video) is a link to the code.


---

### 2. Identify potential shortcomings with your current pipeline

There are lots of potential shortcoming with my current pipeline. Here are a few of the main ones:

1. Lane lines can be hard to detect under different conditions (e.g. lane lines change color, shadows on road, inconsistent pavement color, reflection of sun on road, etc.)
2. The pipeline only works on relatively flat and straight sections of road. It cannot detect or draw lane lines when the road curves.
3. The 'region of interest' dimensions are hard coded pixel values, so you can't use the pipeline on different screen resolutions without manually adjusting the vertices each time.

---

### 3. Suggest possible improvements to your pipeline

Some possible improvements would be:

1. Leverage various color spaces, channels, and filters to identify lane lines more precisely under different light, weather, and road conditions (instead of just using grayscale).
1. Leverage other edge detection methods (other than Canny).
1. Leverage the fact that lane lines are parallel to improve their detection.
1. Use quadratic equations to detect and draw curved lane boundaries (instead of being limited to detecting/drawing straight lines using linear equations).  
1. Define the 'region of interest' based on screen percentage (not pixels) so you can use the algorithm on different screen resolutions.




---
# Project Setup

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).The code file is called P1.ipynb and the writeup template is writeup_template.md

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)


Creating a Great Writeup
---
For this project, a great writeup should provide a detailed response to the "Reflection" section of the [project rubric](https://review.udacity.com/#!/rubrics/322/view). There are three parts to the reflection:

1. Describe the pipeline

2. Identify any shortcomings

3. Suggest possible improvements

We encourage using images in your writeup to demonstrate how your pipeline works.  

All that said, please be concise!  We're not looking for you to write a book here: just a brief description.

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup. Here is a link to a [writeup template file](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).


## Getting Started

If you have already installed the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) you should be good to go! If not, you should install the starter kit to get started on this project. ##

**Step 1:** Set up the [CarND Term1 Starter Kit](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/83ec35ee-1e02-48a5-bdb7-d244bd47c2dc/lessons/8c82408b-a217-4d09-b81d-1bda4c6380ef/concepts/4f1870e0-3849-43e4-b670-12e6f2d4b7a7) if you haven't already.

**Step 2:** Open the code in a Jupyter Notebook

You will complete the project code in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out <A HREF="https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python" target="_blank">Cyrille Rossant's Basics of Jupyter Notebook and Python</A> to get started.

Jupyter is an Ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, use terminal to navigate to your project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) installation instructions!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  

**Step 3:** Complete the project and submit both the Ipython notebook and the project writeup
