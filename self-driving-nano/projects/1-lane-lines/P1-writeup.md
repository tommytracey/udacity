# **Finding Lane Lines on the Road** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consists of five steps:

1. First, I convert the images to grayscale using OpenCV. 

![step1][results/1-grayscale.png]

![step1][results/1-code.png]


2. I then apply Gaussian smoothing. This reduces the amount of noise in the image and makes it easier to identify the most prominent edges in the next step. 

![step2][results/2-gaussian.png]

![step2][results/2-code.png]


3. Next, I apply the Canny transform to identify edges. 

![step3][results/3-canny.png]

![step3][results/3-code.png]


4. Then, I create a mask in the shape of a trapezoid to define the part of the image where we want to detect the lane lines (aka the 'region of interest'). Obviously, we should look for lane lines on the road immediately in front of the car; we should not look for them in the sky. But maybe someday! 8')

![step4][results/4-region-of-interest.png]

![step4][results/4-code.png]


5. Finally, I detect the lane lines using the probabilistic Hough transform method. The lines are then drawn onto a separate image and applied to the original image as a transparency. 

![step5a][results/5a-hough-lines.png]

![step5b][results/5b-final-image.png]


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

![step5][results/5-code.png]




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...





### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

- mark region of interest based on screen percentage (not pixels) so you can use the algorithm on different screen resolutions
- improve lane line detection using different color spaces/filters/channels, since lane lines can be hard to identify in different light conditions (e.g. shadows on road, pavement changes color, etc.)
- use quadratic equations to draw lane boundaries
