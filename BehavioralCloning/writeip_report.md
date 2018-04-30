---
title: "Udacity SDCND - Behavorial Cloning"
author: "Ravi Theja"
date: "20 August 2017"
output: html_document
---

## Project Description

In this project, neural network architecture is used to clone car driving behavior.  It is a supervised regression problem between the car steering angles and the road images in front of a car.  

These images are taken from three different cameras placed in front of the car (one at the center, second one at the left and third one at the right of the car).  

The network is based on [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been proven to work in this problem domain.

As image processing is involved, the model is using convolutional layers for automated feature engineering.  

### Files included

- model.py 
- drive.py 
- utils.py 
- model.h5 
- writeup_report.md
- writeup_report.html
- video.mp4
- relu_error_batch_40.png
- relu_error_batch_50.png

## Quick Start

### Training the model

You'll need the data folder which contains the training images. The following command can be used to run the model.

```python
python model.py
```

### Running the trained model

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), select desired track and press the Autonomous Mode button.  Then, run the model as follows:

```python
python drive.py model.h5
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

The matplotlib plotting functions were used to visualize the MSE loss in training and validation data. The image is named mse.png attached to the files sent.

## Model Architecture

The design of the network is based on [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been used by NVIDIA for the end-to-end self driving test.  

It is a deep convolution network which works well with supervised image classification / regression problems.  As the NVIDIA model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.

I've added the following adjustments to the model. 

- Lambda layer has been used to normalized input images to avoid saturation and make gradients work better.
- An additional dropout layer has been added to avoid overfitting after the convolution layers.
- RELU has been added as an activation function for every layer except for the output layer to introduce non-linearity.

My final model will look like this:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: RELU
- Fully connected: neurons:  50, activation: RELU
- Fully connected: neurons:  10, activation: RELU
- Fully connected: neurons:   1 (output)

As per the NVIDIA model, the convolution layers are meant to handle feature engineering and the fully connected layer for predicting the steering angle.  However, as stated in the NVIDIA document, it is not clear where to draw such a clear distinction.  Overall, the model is very functional to clone the given steering behavior.  

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer (type)                   |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|-------:|-----------------|
|lambda_1 (Lambda)               |(None, 66, 200, 3)|0       |lambda_input_1   |
|convolution2d_1 (Convolution2D) |(None, 31, 98, 24)|1824    |lambda_1         |
|convolution2d_2 (Convolution2D) |(None, 14, 47, 36)|21636   |convolution2d_1  |
|convolution2d_3 (Convolution2D) |(None, 5, 22, 48) |43248   |convolution2d_2  |
|convolution2d_4 (Convolution2D) |(None, 3, 20, 64) |27712   |convolution2d_3  |
|convolution2d_5 (Convolution2D) |(None, 1, 18, 64) |36928   |convolution2d_4  |
|dropout_1 (Dropout)             |(None, 1, 18, 64) |0       |convolution2d_5  |
|flatten_1 (Flatten)             |(None, 1152)      |0       |dropout_1        |
|dense_1 (Dense)                 |(None, 100)       |115300  |flatten_1        |
|dense_2 (Dense)                 |(None, 50)        |5050    |dense_1          |
|dense_3 (Dense)                 |(None, 10)        |510     |dense_2          |
|dense_4 (Dense)                 |(None, 1)         |11      |dense_3          |
|                                |**Total params**  |252219  |                 |


## Data Preprocessing

### Image Sizing

- the images are cropped so that the model won’t be trained with the sky and the car front parts
- the images are resized to 66x200 (3 YUV channels) as per NVIDIA model
- the images are normalized (image data divided by 127.5 and subtracted 1.0).  As stated in the Model Architecture section, this is to avoid saturation and make gradients work better)


## Model Training

### Image Augumentation

For training, I used the following augumentation technique along with Python generator to generate unlimited number of images:

- Randomly choose right, left or center images.
- For left image, steering angle is adjusted by +0.2
- For right image, steering angle is adjusted by -0.2
- Randomly flip image left/right
- Randomly translate image horizontally with steering angle adjustment (0.002 per pixel shift)
- Randomly translate image virtically
- Randomly added shadows
- Randomly altering image brightness (lighter or darker)

Using the left/right images is useful to train the recovery driving scenario.  The horizontal translation is useful for difficult curve handling (i.e. the one after the bridge).


### Examples of Image Transformations

The following is the example transformations:

**Original Image**

![Original Image](processedImages/originalImage.png)

**Cropped Image**

![Cropped Image](processedImages/croppedImage.png)

**YUV Image**

![YUV Image](processedImages/yuv_image.png)

**Flipped Image**

![Flipped Image](processedImages/flipped_image.png)

**Shadow Image**

![Shadow Image](processedImages/shadow_image.png)

**Translated Image**

![Translated Image](processedImages/translated_image.png)

**Brightness Image**

![Brightness Image](processedImages/bright_image.png)

## Training, Validation and Test

Data has been splitted into training and validation set in order to measure the performance at every epoch.  Testing was done using the simulator.

As for training, 

- Mean Squared Error (MSE) has been used as the loss function to measure how close the model predicts the steering angle given an image.
- Adam optimizer has been used as the optimization function with learning rate of 1.0e-4 which is smaller than the default of 1.0e-3.  The default value was too big and made the validation loss stop improving soon.
- ModelCheckpoint from Keras has been used to save the model only if the validation loss is improved which is checked for every epoch.

### The Lake Side Track

I drive the car three times clockwise round the track and three times counter clokwise round the track and collected decent amount of training data. Initially the simulator was running very slow in my laptop and after reading the forum I selected graphics quality as Fastest and collected the training data.  
As there can be unlimited number of images augmented, I set the samples per epoch to 20,000.  I tried from 1 to 15 epochs but I found 5-10 epochs is good enough to produce a well trained model for the lake side track.  The batch size of 50 was chosen.

### The Jungle Track
I have not trained my model on the Jungle track due some time constraints. I want to do it for this little complex jungle track as well whenever I get time.


## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
