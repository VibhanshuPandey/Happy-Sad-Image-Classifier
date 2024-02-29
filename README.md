# Happy/Sad Face Classifier

This project trains a convolutional neural network (CNN) model to classify if faces in images look happy or sad. The model is trained on a dataset of labeled happy and sad face images.

# For downloading all the images from the Google use this extension
https://download-all-images.mobilefirst.me
## Overview

What the model does:

- Takes an image of a face as input
- Outputs a prediction of whether the face looks "happy" or "sad"

It learns to make these predictions by looking at many example happy and sad faces during training.

## Getting Started

### Install dependencies

You'll need Python and some machine learning packages installed:

- TensorFlow - Used to build and train the neural network model
- OpenCV - For image processing
- Matplotlib - For visualizing images and plots
- Numpy - For numerical processing

Install them with `pip`:

```
pip install tensorflow opencv-python matplotlib numpy
``` 

A GPU can speed up training, so install `tensorflow-gpu` instead of `tensorflow` if you have an Nvidia GPU.

### Prepare dataset

The model needs example data to learn from, in the form of labeled happy and sad face images.

Organize your training images into folders:

- `facedata/happy` - Contains images of happy faces
- `facedata/sad` - Contains images of sad faces

The more balanced data you have, the better the model will perform. 

### Train the model

Run `python ml_expt_5.py` to train the model.

This will:

- Load the image data
- Preprocess and split data into training and validation sets 
- Build a CNN model architecture
- Train the CNN on the data by adjusting weights to minimize loss 
- Evaluate accuracy on validation set
- Save the trained model to `models/happysadmodel.h5`

### Make predictions

To classify a new test image, you can load the saved model and call `model.predict(img)`:

```python
from tensorflow.keras.models import load_model

model = load_model('models/happysadmodel.h5')

img = load_and_preprocess('test.jpg')
pred = model.predict(img) 
# Prediction will be happy or sad
```

Where `load_and_preprocess()` handles loading the image and scaling pixel values.

## Model Details

The model architecture uses convolutional and pooling layers to identify facial features and patterns in the images. Some key aspects:

- Uses ReLU activation functions for non-linearity
- Max pooling reduces spatial dimensions after convolutions 
- Flattening converts image data into a 1D vector 
- Fully connected layers classify the features
- Sigmoid output layer gives a probability of happy vs sad

The model is trained for 25 epochs using the binary cross-entropy loss function and Adam optimizer.
