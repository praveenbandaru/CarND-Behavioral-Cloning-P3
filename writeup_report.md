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

[image1]: ./examples/input_augmented_labels.png "Image Augmentation"
[image2]: ./examples/model_summary.png "Model Summary"
[image3]: ./examples/model.png "Model"
[image4]: ./examples/modelwithshapes.png "Model with Shapes"
[image5]: ./examples/model.svg "Model SVG"
[image6]: ./examples/center_image.png "Normal Image"
[image7]: ./examples/flipped_image.png "Flipped Image"
[image8]: ./examples/loss.png "Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 a video recording of the vehicle driving autonomously at least one lap around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 & 3x3 filter sizes and depths between 24 and 64 (model.py lines 78-90) 

The model includes RELU layers to introduce nonlinearity (code lines 81-85), and the data is normalized in the model using a Keras lambda layer (code line 79). 

The model also consists of a Keras cropping layer which trims the image to only see a section containing road by removing sky and hood at the top and bottom of the image respectively (code line 80).

Here is the model summary:
![alt text][image2]

#### 2. Attempts to reduce overfitting in the model

The model doesn't contains any dropout layers or any regularization techniques. 

The model was trained and validated on an augmentated data set containing different data sets from both track1 and track2 to ensure that the model was generalized and not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also I have augmented the data to contain the horizontally flipped versions of all center, left and right images. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with basic known models and then experiment by increasing the number or depth of the convolutional layers.

My first step was to use a convolution neural network model similar to the [LeNet](http://yann.lecun.com/exdb/lenet/ "LeNet") which I've used in my [CarND-Traffic-Sign-Classifier-Project](https://github.com/praveenbandaru/CarND-Traffic-Sign-Classifier-Project "CarND-Traffic-Sign-Classifier-Project").
 I thought this model might be appropriate to begin with because it performed great on classification task for road sign images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by introducing dropout layers and the mean squared error was reduced.

But when I ran the simulator, the car drove well until it reached the bridge on track1 and it crashed.

Then I implemented the convolutional neural network model from [NVIDIA's article](https://devblogs.nvidia.com/deep-learning-self-driving-cars/ "NVIDIA's article").

After training on the new model, the car was able to successfully drive around the track1 without any problem but failed on track2. Then I managed to collect and add more training data for track2 and retrained the model on the final dataset.

At the end of the process, the vehicle is able to drive autonomously around both the tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 78-90) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image5]

#### 3. Creation of the Training Set & Training Process

Initially I started with the [Sample Training Data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip "Sample Training Data") provided by Udacity under Project Resources. To generalize the dataset for both tracks, I recorded three laps on track two using a game controller and combined both the data sets.

The data set consists of images from center, left and right cameras.

Here is an example image from center camera:

![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would help to generalize the dataset for both left and right turns on both tracks.

Here is an comparison of orignal and flipped versions of images from center, left and right cameras captured at the same time:

![alt text][image1]

After the collection process, I had 76674 number of data points. I then preprocessed this data by normalizing and cropping using Keras Lambda and Cropping layers respectively within the model itself.

The code in model uses a Python generator to generate data for training rather than storing the training data in memory.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by loss vs epoch graph. I can run it for more epochs by decreasing the learning rate to get a better mean squared error loss but I haven't pursued as my model was running successfully on the simulator. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image8]
