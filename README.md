# CIFAR-10 Image Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 color images of size 32x32 pixels, grouped into 10 different classes such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Project Overview

The goal of this project is to classify images into their respective categories using a deep learning model. The model is built using TensorFlow and Keras and trained on the CIFAR-10 dataset.

- **Dataset**: CIFAR-10
- **Model**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow and Keras
- **Deployment**: Streamlit app on Hugging Face Spaces

### CIFAR-10 Classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Details:

- **Training Images**: 50,000
- **Test Images**: 10,000
- **Image Dimensions**: 32x32 pixels

## How It Works

1. **Data Loading**: The CIFAR-10 dataset is loaded using the `tensorflow.keras.datasets` module, which provides easy access to the dataset and splits it into training and testing sets.
2. **Model Creation**: A CNN is built using Keras Sequential API, designed to extract spatial features from the input images through convolution and pooling layers.
3. **Training**: The model is trained using the training set and validated on the test set.
4. **Evaluation**: The model is evaluated using accuracy metrics and confusion matrix to understand the classification performance across the different categories.
5. **Deployment**: The trained model is deployed on a Streamlit web application, hosted on Hugging Face Spaces, allowing users to upload images for classification in real time.

## Technology Stack

- **Python**: Programming language used for data processing, model building, and evaluation.
- **TensorFlow & Keras**: Deep learning frameworks used to build and train the CNN model.
- **Streamlit**: Used to create the web interface for deploying the model.
- **Hugging Face Spaces**: Hosting platform for the Streamlit app.

## About the App

The app, hosted on Hugging Face Spaces, allows users to interact with the model in real-time. Users can upload an image and the app will classify it into one of the 10 classes from the CIFAR-10 dataset.

Check out the app on [Hugging Face Spaces](https://huggingface.co/spaces/zafermbilen/cnn-for-cifar10).

---

Feel free to clone this repository and explore the CIFAR-10 classification model!
