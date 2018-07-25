# Neural-Net for classification of Iris dataset::

Hi There,
This is a simple neural net for classification of iris flower dataset.
Iris flower dataset consists of four different features of a flower which can belong to anyone of the 3 classes which are::
  1) setosa.
  2) virginica.
  3) versicolor.

to know more about Iris Dataset refer the link:: https://en.wikipedia.org/wiki/Iris_flower_data_set

The Architecture::
  There are 3 layers from which 1st layer is input layer consisting of four neurons. The middle and also the hidden layer consists          of 100 neurons and the last and the final output layer has 3 neurons. I have used Stochastic gradient descent, ie trained over 120 examples at a time in a vectorized manner. 
  
the activations from 1st to 2ad layer is relu and in 2ad-3rd layer is softmax.
  
This code is written in python 3.6   
  
By running this code output will be accuracy of the trained model over testing data ie test_iris.csv and the graph of Cost function  vs no of iteration.
