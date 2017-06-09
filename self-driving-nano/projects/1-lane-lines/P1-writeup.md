# **Finding Lane Lines on the Road** 

## Writeup


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

1. First, I convert the images to grayscale. 

<img src='results/1-grayscale.png' />

<img src='results/1-code.png' width="100%"/>



2. I then apply Gaussian smoothing. This reduces the amount of noise in the image and makes it easier to identify the most prominent edges in the next step. 

<img src='results/2-gaussian.png' />

<img src='results/2-code.png' width="100%"/>


3. Next, I apply the Canny transform to identify edges within the image. Edges are areas with the strongest gradient -- i.e., areas where there's a large difference in brightness between adjacent pixels. In photography, you might think of these as the areas with the highest contrast. These edges allow us to detect the boundaries of objects within the image. In this case, we're looking for the boundaries of the brighter driving lane lines against the darker background of the highway pavement (but, this process will also detect the edges of other objects). The Canny algorithm allows us to identify the individual pixels where these edges are the strongest.

<img src='results/3-canny.png' />

<img src='results/3-code.png' width="100%"/>


4. Then, I create a mask in the shape of a trapezoid to define the part of the image where we want to detect the lane lines (aka the 'region of interest'). Obviously, we should look for lane lines on the road immediately in front of the car; we should not look for them in the sky. But maybe someday! 8')

<img src='results/4-region-of-interest.png' />

<img src='results/4-code.png' width="100%"/>


5. Finally, I detect the lane lines using the probabilistic Hough transform method. The lines are then drawn onto a separate image and applied to the original image as a transparency. 

<img src='results/5a-hough-lines.png' />

<img src='results/5b-final-image.png' />


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

<img src='results/5-code.png' width="100%"/>




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...





### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

- mark region of interest based on screen percentage (not pixels) so you can use the algorithm on different screen resolutions
- improve lane line detection using different color spaces/filters/channels, since lane lines can be hard to identify in different light conditions (e.g. shadows on road, pavement changes color, etc.)
- use quadratic equations to draw lane boundaries
