---
title: "Neural Networks"
teaching: 20
exercises: 30
questions:
- "How can we classify images using a neural network?"
objectives:
- "Explain the basic architecture of a perceptron."
- "Create a perceptron to encode a simple function."
- "Understand that a single perceptron cannot solve a problem requiring non-linear separability."
- "Understand that layers of perceptrons allow non-linear separable problems to be solved."
- "Evaluate the accuracy of a multi-layer perceptron using real input data."
- "Understand that cross validation allows the entire data set to be used in the training process."
keypoints:
- "Perceptrons are artificial neurons which build neural networks."
- "A perceptron takes multiple inputs, multiplies each by a weight value and sums the weighted inputs. It then applies an activation function to the sum."
- "A single perceptron can solve simple functions which are linearly separable."
- "Multiple perceptrons can be combined to form a neural network which can solve functions that aren't linearly separable."
- "Training a neural network requires some training data to show the network examples of what to learn."
- "To validate our training we split the the training data into a training set and a test set."
- "To ensure the whole dataset can be used in training and testing we can train multiple times with different subsets of the data acting as training/testing data. This is called cross validation."
- "Several companies now offer cloud APIs where we can train neural networks on powerful computers."
---


# Introduction

Neural networks, drawing inspiration from the workings of the human brain, represent a machine learning approach adept at discerning patterns and categorizing data, frequently leveraging images as input. This technique, rooted in the 1950s, has evolved through successive iterations, surmounting inherent constraints. Today, the pinnacle of neural network advancement is often denoted as deep learning.


## Perceptrons

Perceptrons serve as the foundational units within neural networks, mirroring the functionality of individual neurons in the brain. Typically equipped with one or more inputs and a solitary output, they operate by weighting each input and aggregating these weighted values. Subsequently, the summed result undergoes evaluation by an activation function, determining whether the neuron emits a signal. While some activation functions employ a straightforward threshold step mechanism, delineating between zero and one based on input magnitude, alternative designs may utilize different functions. Nevertheless, these functions commonly yield outputs ranging from zero to one and retain a step-wise characteristic.

![A diagram of a perceptron](../fig/perceptron.svg)

### Coding a perceptron

The function requires three parameters: Inputs, a list of input values; Weights, a list of weight values; and Threshold, denoting the activation threshold.
Initially, we perform element-wise multiplication of each input with its corresponding weight.
Subsequently, the total sum of these products is computed. If this sum falls below the activation threshold, the output is zero; otherwise, it is one.


### Perceptron limitations

A solitary perceptron is incapable of resolving any function that lacks linear separability, necessitating the ability to partition input and output classes with a straight line. An illustrative instance is the XOR function below:

| Input 1 | Input 2 | Output |
| --------|---------|--------|
| 0       |0        |0       |
| 0       |1        |1       |
| 1       |0        |1       |
| 1       |1        |0       |

(Make a graph of this)

which yields zero output when all inputs are either one or zero, defying straightforward linear separation. This inadequacy, termed linear separability, was recognized in the 1960s, leading to a stagnation in neural network advancement for over a decade, often referred to as the "AI Winter."


## Multi-layer Perceptrons

A single perceptron lacks the capability to address functions that lack linear separability. To tackle such nonlinear challenges, we rely on multiple perceptrons, often organized into several layers. These layers constitute networks of artificial neurons, each capable of processing one or more inputs and producing a single output. The neurons interconnect within expansive networks, commonly comprising tens to thousands of units. Typically, these networks are structured in layers, encompassing an input layer, one or more hidden layers, and ultimately, an output layer.

![A multi-layer perceptron](../fig/multilayer_perceptron.svg)

### Training Multi-layer perceptrons

Multi-layer perceptrons need to be trained by showing them a set of training data and measuring the error between the network's predicted output and the true value. Training takes an iterative approach that improves the network a little each time a new training example is presented. There are a number of training algorithms available for a neural network today, but we are going to use one of the best established and well known, the backpropagation algorithm. The algorithm is called back propagation because it takes the error calculated between an output of the network and the true value and takes it back through the network to update the weights. If you want to read more about back propagation, please see [this chapter](http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf) from the book "Neural Networks - A Systematic Introduction".

### Multi-layer perceptrons training in R

We're preparing to construct a multi-layer perceptron to predict species in the iris dataset. With the dataset's four computed features representing two aspects of the plant, along with width and height, we'll set up four input neurons. The number of hidden layers can vary and is typically determined through experimentation. Since there are three different species of plants, we'll incorporate three output neurons.

Before delving into the construction, let's organize our data for ingestion into the neural network:

~~~
### !pip install tensorflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

import tensorflow as tf



iris_df = pd.read_csv("iris.csv")


lbl_clf = LabelEncoder()
Y_encoded = lbl_clf.fit_transform(iris_df['variety'])

#Keras requires your output feature to be one-hot encoded values.
Y_final = tf.keras.utils.to_categorical(Y_encoded)


X_train, X_test, y_train, y_test = train_test_split(iris_df.iloc[:, :4], Y_final, test_size = 0.25, random_state = 0)


std_clf = StandardScaler()
x_train_new = std_clf.fit_transform(X_train)
x_test_new = std_clf.transform(X_test)
print(x_train_new)
print(y_train)
~~~
{: .language-python}

Now lets build our neural network, to which we use a library called neuralnet:
~~~
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_dim=4, activation=tf.nn.relu, kernel_initializer='he_normal', 
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(7, activation=tf.nn.relu, kernel_initializer='he_normal', 
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu, kernel_initializer='he_normal', 
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

iris_model = model.fit(x_train_new, y_train, epochs=10, batch_size=7)
~~~
{: .language-python}

><pre style="color: black; background: white;">
>Epoch 1/5
>16/16 [==============================] - 1s 2ms/step - loss: 1.5008 - accuracy: 0.3929
>Epoch 2/5
>16/16 [==============================] - 0s 2ms/step - loss: 1.4058 - accuracy: 0.3929
>Epoch 3/5
>16/16 [==============================] - 0s 2ms/step - loss: 1.3837 - accuracy: 0.3929
>Epoch 4/5
>16/16 [==============================] - 0s 2ms/step - loss: 1.2541 - accuracy: 0.4911
>Epoch 5/5
>16/16 [==============================] - 0s 2ms/step - loss: 1.2644 - accuracy: 0.4911
></pre>
{: .output}


## Prediction using a multi-layer perceptron
### Confusion Matrix

To look at how our model performed, there are a number of ways you could look at it. The best way is to have look at the confusion matrix and luckily in R there is a built in function that does this for us. All we have to do is pass our prediction results to the table function. Furthermore, by summing the diagonal and dividing by the length of our test set we can come up with an accuracy value. 

~~~
import sklearn.metrics as skm
model.evaluate(x_test_new, y_test)


results = model.predict(x_test_new)

def convert_labels(array):
    collection = []
    for ii in range(array.shape[0]):
        val = np.argmax(array[ii,:])
        lab = np.zeros((1, array.shape[1]))
        lab[:,val] = 1
        collection.append(lab)
        
    collection = np.array(collection)
    collection = np.squeeze(collection)
    return collection


argmax_results = convert_labels(results)
CON_MATRIX = skm.multilabel_confusion_matrix(y_test, argmax_results)
print(CON_MATRIX)
~~~
{: .language-python}



~~~
[[[25  0]
  [13  0]]

 [[ 9 13]
  [16  0]]

 [[13 16]
  [ 0  9]]]
~~~
{: .output}


> ## Changing the different characteristics of the neural network
> There are a number of characteristics you can change in your model, that may increase or decrease the performance of your model. have ago at adjusting the number of steps, linear output ("T" or "F") and number of hidden layers.
{: .challenge}



### Cloud APIs

Google, Microsoft, Amazon, and many others now have Cloud based Application Programming Interfaces (APIs) where you can upload an image and have them return you the result. Most of these services rely on a large pre-trained (and often proprietary) neural network.

> ## Exercise: Try cloud image classification
> Take a photo with your phone camera or find an image online of a common daily scene.
> Upload it Google's Vision AI example at https://cloud.google.com/vision/
> How many objects has it correctly classified? How many did it incorrectly classify?
> Try the same image with Microsoft's Computer Vision API at https://azure.microsoft.com/en-gb/services/cognitive-services/computer-vision/
> Does it do any better/worse than Google?
{: .challenge}

### Existing API's of machine learning models

A vast collection of deep learning machine models can be found on the platform known as Hugging Face. Serving as an AI community hub, it offers a diverse array of pre-trained, cutting-edge machine learning models accessible to all users.

> ## Exercise: Existing API's of machine learning models
> go to https://huggingface.co/ and have ago at some of the different models, alot of them have inference API so you can have ago on the website. 
>
{: .challenge}

{% include links.md %}
