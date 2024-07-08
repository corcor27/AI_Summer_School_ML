---
title: "Non-Linear Classifiers"
teaching: 15
exercises: 20
questions:
- "How can I process data?"
objectives:
- "Recall how to build non-linear models."
- "Understand that more complex models can be built with non-linear equations."
- "To be able to predict using non-linear models"
keypoints:
- "Learning powerful library's to implement machine learning functions."
- "Used non-linear machine learning models to predict results"
---


## K-Nearest Neighbour (KNN)

The k-nearest Neighbours algorithm, commonly referred to as KNN or k-NN, is a supervised learning classifier that falls under the non-parametric category. It leverages proximity to classify or predict the grouping of a specific data point. Although it can tackle both regression and classification tasks, it is predominantly employed as a classification tool. The underlying principle is based on the assumption that similar data points tend to cluster together.
In classification scenarios, the algorithm assigns a class label through a majority vote mechanism. In other words, the label that appears most frequently among neighboring data points is adopted. While technically termed "plurality voting," it is often referred to as "majority vote" in literature. The distinction lies in the requirement for a true majority (over 50%), which suits binary classification situations. In cases involving multiple classes (e.g., four categories), a conclusive decision regarding a class label can be made with a threshold vote exceeding 25%.

Before we train any non-linear machine learning models, we need to divide our data into train and test sets. To do this we use a library called scikit learn. 
Furthermore, traditionally machine learning models only accept inputs which are between zero and one. so we will also need to scale our data.  

~~~
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split



iris_df = pd.read_csv("iris.csv")


iris_df['labels'] = iris_df.variety.astype('category').cat.codes

X, y = iris_df.iloc[:, :4], iris_df['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
clf = neighbors.KNeighborsClassifier(n_neighbors=5)

clf.fit(X_train, y_train)
~~~
{: .language-python}


Now lets build our self KNN model with scikit learn.

~~~
result = clf.predict(X_test)
ground = np.array(y_test)
print(result)

~~~
{: .language-python}

><pre style="color: black; background: white;">
>[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
> 2]
></pre>
{: .output}

### Confusion Matrix

To look at how our model performed, there are a number of ways you could look at it. The best way is to have look at the confusion matrix and luckily in R there is a built in function that does this for us. All we have to do is pass our prediction results to the table function. Furthermore, by summing the diagonal and dividing by the length of our test set we can come up with an accuracy value. 

~~~
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


matrix = confusion_matrix(ground, result)#confusion_matrix(truth, prediction)
print(matrix)

count = 0 

for row in range(0, y_test.shape[0]):
    if result[row] == ground[row]:
        count += 1
print((count/y_test.shape[0])*100)

sp = iris_df.drop_duplicates(subset=['variety'])
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=list(sp['variety']))
disp.plot()
~~~
{: .language-python}



~~~
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
97.36842105263158
~~~
{: .output}

## Support Vector Machines (SVM)

The Support Vector Machine (SVM) emerges as a formidable supervised algorithm, demonstrating its effectiveness particularly on smaller yet intricate datasets. While adept at handling both regression and classification tasks, SVMs notably shine in classification scenarios. Originating in the 1990s, SVMs garnered widespread recognition and endure as a favoured option for high-performance algorithms, often requiring minimal adjustments to yield robust outcomes. Described as a machine learning algorithm utilising supervised learning models, SVMs tackle intricate classification, regression, and outlier detection challenges by executing optimal data transformations. These transformations delineate boundaries between data points based on predefined classes, labels, or outputs. This article elucidates the core principles of SVMs, their functionality, variations, and offers insights through real-world illustrations.

### Strengths of support vector machines:

- Effective in navigating high-dimensional spaces.
- Remain potent even when faced with a higher number of dimensions compared to samples.
- Operate efficiently on memory by utilizing a subset of training points known as support vectors in the decision-making process.
- Offer versatility through the option to specify various Kernel functions for the decision function, including the provision for custom kernels.

### Drawbacks of support vector machines:

- When the number of features significantly exceeds the number of samples, guarding against over-fitting necessitates careful selection of Kernel functions and regularization terms.
- Direct probability estimates are not provided by SVMs; obtaining such estimates involves resource-intensive techniques like five-fold cross-validation (refer to Scores and probabilities).

### SVM in Python

So to create a SVM model, we are going to use the library called scikitlearn. We are also going to use our train/test separations from above.

~~~
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split



iris_df = pd.read_csv("iris.csv")


iris_df['labels'] = iris_df.variety.astype('category').cat.codes

X, y = iris_df.iloc[:, :4], iris_df['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
clf = SVC(kernel = 'poly', random_state = 0)

clf.fit(X_train, y_train)
~~~
{: .language-python}

Now lets have ago at predicting our test set using the SVM model. Again we are going to produce a confusion matrix and generate an accuracy score.

~~~
result = clf.predict(X_test)
ground = np.array(y_test)
print(result)

from sklearn.metrics import confusion_matrix


matrix = confusion_matrix(ground, result)#confusion_matrix(truth, prediction)
print(matrix)

count = 0 

for row in range(0, y_test.shape[0]):
    if result[row] == ground[row]:
        count += 1
print((count/y_test.shape[0])*100)
~~~
{: .language-python}
~~~
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
97.36842105263158
~~~
{: .output}

> ## different non-linear classifier
>
> Have ago at implementing a different non-linear classifier. examples of decision tree can be found at: https://www.datacamp.com/tutorial/decision-tree-classification-python
> Or even Random forest: https://www.datacamp.com/tutorial/random-forests-classifier-python
{: .challenge}



{% include links.md %}
