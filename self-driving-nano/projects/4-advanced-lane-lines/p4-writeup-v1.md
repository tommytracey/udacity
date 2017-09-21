**Udacity Nanodegree &nbsp; | &nbsp; Self-Driving Car Engineer**
# Project 4: Advanced Lane Detection

### Goals
The goals of this project are to:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The original Udacity template and source code for this project can be found [here in their Github repo](https://github.com/udacity/CarND-Advanced-Lane-Lines).

&nbsp;
### Results
For this project, I was able to accurately detect the driving lane and project it back onto the input from the vehicle's camera as required. Here is a video that shows the results.  

&nbsp;

<a href="https://www.youtube.com/embed/rJieV8ADRq4?rel=0" target="_blank"><img src="results/track1.png" width="60%" /></a>


&nbsp;
### My Approach
You can find a step-by-step breakdown of my approach and the various parts of my pipeline [here in this Jupyter notebook](). In the next section, I will outline how I addressed the required aspects of this project.


&nbsp;
---
## Rubric Points
In this section, I walk-through the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.

&nbsp;
### Camera Calibration

&nbsp;
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

[(source code)]()

<img src='results/ABCDE.png' width="60%"/>

&nbsp;
### Pipeline (single images)

&nbsp;
#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

<img src='results/ABCDE.png' width="60%"/>

&nbsp;
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

[(source code)]()

<img src='results/ABCDE.png' width="60%"/>

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

[(source code)]()

<img src='results/ABCDE.png' width="60%"/>

&nbsp;
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

[(source code)]()

<img src='results/ABCDE.png' width="60%"/>

&nbsp;
```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```
&nbsp;

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

&nbsp;

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

[(source code)]()

<img src='results/ABCDE.png' width="60%"/>

&nbsp;

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

[(source code)]()

<img src='results/ABCDE.png' width="60%"/>

&nbsp;

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

[(source code)]()

<img src='results/ABCDE.png' width="60%"/>

&nbsp;

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

[(source code)]()

<img src='results/ABCDE.png' width="60%"/>

&nbsp;

---

### Discussion

&nbsp;
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
