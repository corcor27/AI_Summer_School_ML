---
title: "Introduction to machine learning"
teaching: 30
exercises: 10
questions:
- What is machine learning?
objectives:
- "Gain an overview of what machine learning is."
- "Understand how machine learning and artificial intelligence differ."
- "Understand some common examples of machine learning being used in our daily lives"
keypoints:
- "Machine learning is a set of tools and techniques to find patterns in data."
- "Some machine learning techniques are useful for predicting something given some input data."
- "Some machine learning techniques are useful for classifying input data and working out which class it belongs to."
- "Artificial Intelligence is a broader term that refers to making computers show human like intelligence."
- "Some people say Artificial Intelligence to mean machine learning"
- "All machine learning systems have some kinds of limitations"
---

# What is machine learning?

Machine learning comprises a variety of tools and methodologies designed to uncover patterns within datasets. This lesson aims to introduce a selection of these techniques, although there exist numerous others beyond the scope of this session.
These techniques can be broadly categorised into two main groups: predictors and classifiers. Predictors are employed to forecast a value or a set of values based on a given set of inputs. For instance, they may predict the cost of an item considering economic conditions and the price of raw materials, or forecast a country's GDP based on its life expectancy. On the other hand, classifiers are tasked with categorised data into distinct groups. For example, they might discern visible characters within an image of written text, or determine whether a message is spam or legitimate.


## Training Data

Many machine learning systems, although not all, acquire knowledge by processing a sequence of input and output data, which they then utilize to construct a model. The mathematical underpinnings of machine learning are agnostic to the nature of the data, as long as it can be represented numerically or categorised. Examples of such applications include:

* Estimating an individual's weight based on their height
* Predicting commute durations given prevailing traffic conditions
* Forecasting housing prices based on stock market fluctuations
* Distinguishing between spam and legitimate emails
* Identifying whether an image contains a person or not


Typically, these models require extensive training with hundreds, thousands, or even millions of examples before they achieve sufficient accuracy for practical predictions or classifications.
Some systems undertake training as a one-time process, resulting in the creation of a model. Others may continuously refine their training through real-world system usage and human feedback also know as reinforcement learning. For instance, every time a user labels an email as spam or not spam, they likely contribute to further training of the spam filter's model.

### Types of output

Predictors will usually involve a continuous scale of outputs, such as the price of something. Classifiers will tell you which class (or classes) are present in the data. For example a system to recognise hand writing from an input image will need to classify the output into one of a set of potential characters. 


## Machine learning vs Artificial Intelligence

Artificial Intelligence encompasses systems with generalized intelligence, theoretically capable of solving a wide array of problems. However, AI is a broad term with varying interpretations. Machine learning systems, on the other hand, are typically trained to address specific problems. While they may exhibit learning behaviour, they lack the generalized intelligence to solve any problem a human could tackle. These systems often require hundreds or thousands of examples to learn and are limited to relatively straightforward classifications. In contrast, a human-like system could learn from a single example.
Another definition of Artificial Intelligence traces back to the 1950s and Alan Turing's "Imitation Game." According to this concept, a system could be deemed intelligent if it could deceive a human into believing they were interacting with another human when in fact, they were conversing with a computer. Modern endeavours in this realm are approaching the point of successfully fooling humans, yet achieving a machine with full human-like intelligence remains a distant prospect.

# Applications of machine learning

## Machine learning in our daily lives

 * Image Recognition
 * Object Detection
 * Character Recognition 
 * Insurance Premiums
 * Energy usage


## Example of machine learning in research
 * Detecting water leaks in pipes.
 * Cancer detection.
 * Improving farming productivity.



# Limitations of Machine Learning

## Garbage In = Garbage Out

In Computer Science, there's a well-known saying: "Garbage In = Garbage Out." This adage highlights the principle that if the input data provided is of poor quality or irrelevant, the resulting output will likely be similarly flawed. For example, if we attempt to train a machine learning system to establish a correlation between two variables that are fundamentally unrelated, the model may still generate a semblance of a connection, but the output will lack meaningful significance. This is often apparent when the model's output appears erratic or seemingly random.

## Bias or lacking training data

The input data may also lack sufficient diversity to encompass all potential scenarios. Biases present in the data collection process can subsequently manifest in the machine learning system. For instance, if data on crime reporting is gathered, it may skew towards wealthier areas where incidents are more likely to be reported. Historical data might be inadequate in terms of coverage or relevance to the specific context being analysed. For example, imagine creating a model to transcribe written text from historical documents. If the model is trained solely on documents from the 1950s to 2000, it may perform well when tested on similar samples from that era. However, testing the model on pre-1950s material might yield poor results because handwriting styles and language usage evolve over time.

## Extrapolation

We can only confidently forecast outcomes for data that falls within the range of our training data. When attempting to extrapolate beyond the scope of our training data, it's likely that our predictions will be inaccurate. An easy way to see this is to plot your training data based on it features along with the sample you want to analyse. If the sample is no where near your data then you could consider this sample an outlier.

## Over fitting

Sometimes ML algorithms become over trained to their training data and struggle to work when presented with real data. Meaning that the model has focused too much on certain characteristics that determine said task, but these may not be applicable when it is used to predict on the test set. This again results in some random predictions. Therefore, its critical not to over train (train for too long) your model. 

## Inability to explain answers

Many machine learning techniques will give us an answer given some input data even if that answer is wrong. Most are unable to explain any kind of logic in arriving at that answer. This can make diagnosing and even detecting problems with them difficult.

{% include links.md %}
