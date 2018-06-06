# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./images/center.jpg
[center_flip]: ./images/center_flip.jpg
[center_crop]: ./images/center_crop.jpg
[right]: ./images/right.jpg
[left]: ./images/left.jpg
[result]: ./images/train_result.jpg


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup.md** which summarizing the results
* **unity_car.ipynb** is identical with **model.py** but increase the readability.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 6 and 64 (model.py lines 91-97) 

The model includes **RELU layers** to introduce nonlinearity after each convolution layer and fully connected layer (except the final layer), and the data is normalized in the model using a Keras lambda layer (code line 90). 

#### 2. Attempts to reduce overfitting in the model

The model **does not** contains dropout layers. Which might be a good improvement direction. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 105). Besides, a data augmentation is used at line 63- 71. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The training data is composed of **center**, **left**, and **right** camera's images.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was referenced with [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

My first step was to use a convolution neural network model similar to the LetNet. I thought this model might be appropriate because the training images contains very basic features like edges of road and line markers. From the experience, we knew that convolutionl layer has good ability to extract/detect these basic features well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the more convolution layers are used.

Then I augmented the data by simpliy flip the images (corresponding with the **flip** steering angles) 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to handle the siutation of left/right departure.
The following images show the left and right perspectives.


![alt text][left]
![alt text][right]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help to eliminate the problemt that **the lap is most of the time turning left!**. For example, here is an image that has then been flipped:

![alt text][center_flip]

In addition, as can be seen in the collected images, the uppder part (trees and sky) and lower part (front car) can be ignored. Since the information might affect the model, therefore, a corp preprocessing is used as show.

![alt text][center_crop]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was  as evidenced by image below I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][result]
