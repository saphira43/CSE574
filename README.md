# CSE574
Final Project for Machine Learning class in University at Buffalo. Fall '15 

This project implements and evaluates classification algorithms. The classification task will be to
recognize a 28 ⇥ 28 grayscale handwritten digit image and identify it as a digit among 0, 1, 2, ... , 9.
The project is implemented in MATLAB.

For the training of our classifiers, we used the MNIST dataset.
The database contains 60,000 training images and 10,000 testing images. It can be
downloaded from here: http://yann.lecun.com/exdb/mnist/
The original black and white (bilevel) images from MNIST were size normalized to fit in a 20x20
pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of
the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28
image by computing the center of mass of the pixels, and translating the image so as to position this
point at the center of the 28x28 field.


1)Logistic Regression
Used SGD.

2)2 Layer neural Network with tanh as the activation function in the first layer and 
  logistic regression as the activation function in the second layer.

