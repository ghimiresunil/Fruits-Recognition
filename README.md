## Fruits Identification Systems

- Dealing with Image datasets.
- Performing Data Processing and Augmentation as and when required.
- Creating and training a Convolutional Neural Network using Tensorflow 2.0.

## Pre-Requisites

- Good Knowledge of Python programming language, as the whole code would be written in python.
- Depth knowledge over Linear and Logistic regression because these are the Bricks of Neural Networks.
- Understanding of Basic Image processing.
- Basic Understanding on Artificial neural networks and Convolutional neural networks Working and Implementation.

## Binary vs Multiclass

- Binary Classification problem statements have two outputs, it would be 1 or 0.
- But in the real world there would be multiple objects and we cannot create a binary classification model for each object to predict whether it's that object or not. 
- Multi class classification addresses this issue.
- In this type of problem we predict a vector of outputs, where each index of vector represents a different class.
- Here is an example of multi class classification:
  - Bike
  - Car
  - Cycle
  - Truck
- Compared to Binary Classification.
- Multi class classification problems are difficult to solve because now there are multiple classes.
- It's easy for our model to commit a mistake between classes.
- In Binary Classification the probability for a random guess to win is 50% but for Multi class it decreases with increase in number of classes.
- Given 131 class classification in our current dataset, the probability for a random guess to win is 0.7% which is almost 70 times less likely compared to Binary class.

## Setup

To run on a normal CPU, having a ‘CUDA’ enabled GPU helps models get trained quicker.

- ‘Google’s Colaboratory’ is the developer's best friend when it comes to deep learning.
- Colaboratory is a Google research project created to help machine learning education and research. 
- Hosted on Google Cloud instances which we can use for free. 

## Real Time Prediction

- We have built a model that can classify different types of fruits and vegetables, we can go ahead and make a simple application out of it.
- A customer would click an image with his smart phone and upload it in our application and we detect the fruit name and return it back to the customer. 
- We have to build a pipeline that takes an input as a single image and the output would be a string that tells the name of the fruit. 

## Summary 

- Google Colab and GPUs.
- Utilizing the Resources for solving High level Computational Problems where Datasets consist of Images, Text, and Videos.
- To Set up the environment for Solving Deep Learning Problems.
- How to use Tensorflow to build Convolutional Neural Networks.
- Created a Model that can detect different types of fruits and vegetables from their images.
- We have trained multiple models and compared their result to select the best performing model. 
- Two phase training on our models, where we have frozen a few layers before our model starts overfitting. 
- Evaluated our model on Test set and applied real time prediction.

## Ways to improve our training results

- Increasing the training in Phase 1 for 1 more epoch.
- Increase Number of phases and Decrease number of epochs per phase. 
- **For example**: You can train a model using a phase 1 with 2 epochs repeated by phase 2 with 4 epochs for 2 times.
- Better version of EfficientNet, as EfficientNetB3 performed better in this case.

## Result
![result](https://user-images.githubusercontent.com/40186859/129464652-a1d32c02-5e13-434e-8a9d-eb025df5b674.png)
