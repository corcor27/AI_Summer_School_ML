---
title: "Time Series Classification"
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

### Materials

All materials for this example can be downloaded with the following link: https://drive.google.com/drive/folders/15UI8jbgp_EOVParKWVtISaBjMgxtUD2v?usp=sharing


{% include links.md %}
