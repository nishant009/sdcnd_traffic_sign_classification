# **Traffic Sign Recognition**

## Writeup Template

---

### **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/train_data_dist.png "Visualization Training"
[image2]: ./writeup_images/valid_data_dist.png "Visualization Validation"
[image3]: ./writeup_images/test_data_dist.png "Visualization Testing"
[image4]: ./writeup_images/grayscale.png "Grayscaling"
[image5]: ./writeup_images/random_noise.png "Random Noise"
[image6]: ./internet_images/1.jpg "Traffic Sign 2"
[image7]: ./internet_images/2.png "Traffic Sign 3"
[image8]: ./internet_images/3.png "Traffic Sign 1"
[image9]: ./internet_images/4.jpg "Traffic Sign 4"
[image10]: ./internet_images/5.jpg "Traffic Sign 6"
[image11]: ./internet_images/6.jpg "Traffic Sign 7"
[image12]: ./internet_images/7.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/nishant009/sdcnd_traffic_sign_classification/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed across the 43 classes. As we can see the distribution is not uniform. Some classes have way more examples than others. This type of training data cause problems with the prediction accuracy of the network.

![alt text][image1]

A similar bar chart of validation and test data below shows the non-uniformity across classes in these data sets as well:

![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because in this particular task the color channels do not provide extra information when it comes to determining the shape of the traffic sign and by extension its class. Color on the other hand only adds more noise. Grayscaling removes this noise and improves efficiency when it comes to prediction.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As a last step, I normalized the image data because the range of values that each pixel can take is large and hence changes in each feature during backpropagation learning would be different. Normalizing pixel values would bring consistency and shorten the range of values each pixel can take.

I decided to generate additional data because I noticed that with the given data set the prediction accuracy of the network was low and it fluctuated wildly. This was because of difference in number of examples per label.

To add more data to the the data set, I generated a set of all such labels that had less than 800 examples. For each label in the set I isolated images in training data with that label. From each such set of isolated images, I randomly selected an image and augmented it. I repeated the random selection and augmentation until that label had 800 examples. This way I added 11681 total images.

Here are some examples of original images and their augmented versions:

![alt text][image5]

The difference between the original data set and the augmented data set is that the images in the augmented data set are:
* Rotated randomly between -10 and 10 degrees
* Translated between -10 and 10 pixels in all directions
* Zoomed between 1 and 1.3
* Sheared between -25 and 25 degrees


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 2x2x100 |
| RELU					|												|
| Flatten				| Output 400 |
| Fully connected		| Input 400 Output 120|
| RELU					|												|
| Dropout					|	 |
| Fully connected		| Input 120 Output 43|
| Softmax				|         									||



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the _AdamOptimizer_ to minimize the mean cross entropy between the softmax output of the logits and the one-hot encoded labels. In addition to that I used the following hyperparameters and values:
* Learning rate = 0.001
* Epochs 35
* Batch size = 135
* Training dropout probability = 0.5
* Distribution mean for weight initialization = 0
* Distribution stddev for weight initialization = 0.1
* Padding = valid
* Convolution stride = 1x1
* Pooling stride = 2x2

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.4%
* test set accuracy of 93.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * I chose to start with the LeNet architecture from the previous lab as that would help me establish a baseline performance and make sure that all the code surrounding the actual NN (preprocessing etc.) was working correctly.
* What were some problems with the initial architecture?
    * Accuracy for the validation set was only 91%
* How was the architecture adjusted and why was it adjusted?
    * I saw that the architecture was overfitting when run for higher number of epochs. I added dropout to the fully connected layers to counter that.
    * Even after that I wasn't getting significant gains in the accuracy. At which point I decided to add yet another layer of convolution to compress features further.
    * Adding a convolution layer helped but I was still seeing a lot of fluctuation in loss across epochs. I decided to remove the penultimate fully connected layer to see if that helped with the loss fluctuation.
    * Reducing the number of fully connected layer helped me get the loss fluctuation in control and get acceptable accuracy.
* Which parameters were tuned? How were they adjusted and why?
    * I also adjusted the batch size to 135 and the number of epochs to 35. This was done mostly via a trail and error approach, finally settling on values that provided some incremental gains on accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Adding dropout layer to the fully connected layer helped me stop the network from overfitting on the dataset.
    * Adding a convolution layer helped me compress the features further, remove noise, and extract more complex shapes. This in turn made the classification task for the fully connected layers easier.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image8] ![alt text][image6] ![alt text][image7]
![alt text][image9] ![alt text][image12] ![alt text][image10] ![alt text][image11]

The fifth image might be difficult to classify because there weren't so many examples for this image type to begin with. After data augmentation 470 additional examples were added for this image type. But it seems that even with augmentation the network was not able to learn the right features for this images type pointing to possible improvements in augmentation techniques.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 70 km/h      		| 70 km/h   									|
| Yield     			| Yield 										|
| Stop     			| Stop 										|
| Road work ahead					| Road work ahead											|
| Straight or right only     		| Road work ahead					 				|
| Turn right ahead			| Turn right ahead      							|
| Right of way       | Right of way      |


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This compares favorably to the accuracy on the test set of 93.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a 70 km/h speed limit sign (probability of 0.867), and the image does contain a 70 km/h speed limit sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .867         			| 70 km/h   									|
| .129     				| 20 km/h 										|
| .003					| 30 km/h											|
| 6.6e-08      	| Road narrows on right					 				|
| 2.4e-08				    | Wild animals crossing      							|

For the second image, the model is absolutely sure that this is a yield sign (probability of 1), and the image does contain a yield sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Yield   									|
| 1.9e-31     				| Turn left ahead 										|
| 5.9e-33					| Stop											|
| 7.3e-34      	| Keep right					 				|
| 1.9e-36				    | Keep left      							|

For the third image, the model is certain that this is a stop sign (probability of 0.995), and the image does contain a stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .995         			| Stop   									|
| .485     				| Yield 										|
| 1.29e-12					| Turn left ahead											|
| 8.23e-18      	| Turn right ahead					 				|
| 4.86e-20				    | Keep right      							|

For the fourth image, the model is absolutely sure that this is a road work ahead sign (probability of 1), and the image does contain a road work ahead sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Road work ahead   									|
| 8.51e-32     				| Wild animals crossing 										|
| 0				| 20 km/h											|
| 0      	| 30 km/h					 				|
| 0				    | 50 km/h      							|

For the fifth image, the model thinks it is a road work ahead sign (probability of 0.524), when the image actually contains a straight or right only sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .524         			| Road work ahead   									|
| .475     				| Turn right ahead 										|
| 9.11e-06					| Stop											|
| 5.33e-10      	| Beware of ice/snow					 				|
| 8.81e-15				    | Turn left ahead      							|

For the sixth image, the model is pretty certain that this is a turn right ahead sign (probability of .999), and the image does contain a turn right ahead sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .999         			| Turn right ahead   									|
| .0001     				| Ahead only 										|
| .00006				| 60 km/h											|
| .00002      	| Road work ahead					 				|
| .00002				    | Turn left ahead      							|

For the seventh image, the model is absolutely sure that this is a right of way sign (probability of 1), and the image does contain a right of way sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Right of way   									|
| 1.78e-16     				| Beware of ice/snow 										|
| 2.35e-29				| Dangerous curve to the right											|
| 3.36e-32      	| 100 km/h					 				|
| 1.57e-33				    | Roundabout mandatory      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
