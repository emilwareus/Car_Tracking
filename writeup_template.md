

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./output_images/feature_image.JPG
[image4]: ./output_images/img_w_boxes.JPG
[image6]: ./examples/labels_map.png
[image7]: ./output_images/img_hot.JPG
[video1]: ./project_video_proc_2.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this headline is under the "Extract Features" header in the notebook Car_Tracking. To extract the HOG features I use the library and function "skimage.feature import hog". The relevant inputs are ' img, orientations, pixels_per_cell and cells_per_block. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:


![alt text][image1]


#### 2. Explain how you settled on your final choice of HOG parameters.

After exploring HOG features in different color spaces I decided to go for the YUV color space, as it seemed to do best in training. I kept iterating with orient, pixels_per_cell and cells_per_block as well and found that 9, 16 and 2 worked best for me. 



![alt text][image2]

.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a function from sklearn. This function is simple to use, but slow. A way to make this faster could be to create a more light, weight classifier. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the function find_car under the Sliding_window header does the sliding window action. It is implemented with help from the exampel given at udacity. It consists of two for-loops that loops over the potential windows in x and y direction. 


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize I only calculate the HOG-features once per image, and not for every subimage in the sliding window. Then I picked 3 different scales of the sliding window. The larger scales are ment for the lower parts of the image and the smaller for the upper part of the image. This is as those cares higher up in the image thend to be further away and there fore smaller. 



![alt text][image4]

![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used a heat map to combine boundingboxes. This was to filter out false positives. I applied a threashold for size as 3000 pixels and a filter forcing the square to be a horizontal rectangle rather than a vertical one.  

The heatmap used "from scipy.ndimage.measurements import label" to devide the heat into different areas to label. I constructed boxes around each blob.  


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had a lot of problems implementing this project. As the HOG features seemed not to be very stable to classify on. I have used Faster-RCNN's in other projects, which have turned out faster and more accurate. It is of course good to know these older technique, but I would like the option to implement something that is a bit more up to date.

I could probably improve the performance by using more features or a different classifier. I was thinking of adding a nother layer of HOG-features with different pix_size and orient. This might help getting cars that are further away. 

