## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 263 through 299 of the file called `car_detection_func.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` 


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the final parameters are:

colorspace = 'YUV'
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size=(16,16)
hist_bins=16

					

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained with a linear SVC.

The code for this step is contained in lines 73 through 76 of the file called `main.py`).  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

in the file car_detection_func.py line 224 to 254


First at all i check the test imgs and decide 4 windows size in the sliding window, that is 64x64, 72x72, 96x96 and 128x128. The large window is to looking for the vehicles near the camera. Actually i have test a lot windows size in the project and decide those 4 windows are good enough.

For the range of the vehicles, i check the vedio and test imgs, and choose the range of the vehicles along with the windows size together.

In the find_cars function, i decide the move_steps of the window to be 2 cells. So the overlap ratio should be 2/64.



![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Since in the course i found that the YUV color space is better for linearSVC so i choose YUV color space. And of course the HOG features, along with the color and spatial features, since they can provide a better result.
But in those features the HOG features should be dominant otherwise the detection will be worse.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out_put_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

in the file car_detection_func.py line 188 to 220

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

In the first phase, the HOG+SVM approach turned out to be slightly frustrating, in that strongly relied on the parameter chosed to perform feature extraction, training and detection. Even if I found a set of parameters that more or less worked for the project video, I wasn't satisfied of the result, because parameters were so finely tuned on the project video that certainly were not robust to different situations.

For this reason, I turned to deep learning.

