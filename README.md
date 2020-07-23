# 2020-07-09
## Framing Key ML Terminology

**Label** - (y) is the thing we are predicting.

**feature** - (x) is an input variable that is used to predict the label.

**Training** - means creating or learning the model.

**Inference** - means applying the trained model to the unlabeled examples.

**Regression Model** - predicts the continuous values
* What is the value of a house?
* What is the probability that a user will click an add?

**Classification Model** - predicts discrete values
* is an email spam?
* is this image a dog or a cat or a hamster?

## Descending into ML
A simple 2d model can be defined with: y1 = b + w1 * x1
* y is the target value
* w is the weight values
* x is the input
* b is the bias. (the y intercept, sometimes referred to as w0)

Some models might depend on more than one input making the equation something like: y1 = b + w1 * x1 + w2 * x2 ..

**Loss** - is how close the predictions are to the model.

**L2 Loss (Squared Loss)** - square of the difference between prediction and label (y - y1)^2
* this amplifies the loss value exponentially as the distance from the prediction increases.

**Emperical Risk Minimization** - In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called empirical risk minimization.

## Reducing Loss
**Gradient Decent** - and algorithm used to find a local minimum for loss.
* **Stochastic** - one example at a time
* **Mini-Batch** - Use small batches and Loss & gradients are averaged over the batch.

A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.

**Convex** - U

**Concave** - W

### How to calculate Gradient Decent
1. Pick a starting value for your weight w1. The starting point doesn't really matter so it can be random or trivial.
2. Calculate the Gradient Loss of your initial value(s). This would be the derivative (slope) of the curve on a graph to tell you which direction is "warmer" or "colder"
   * Note that gradients are vectors so they have both a **direction** and a **magnitude**
   * The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.
3. To calculate the next step by applying some fraction (Learning Rate) of the gradient's magnitude to your starting point value.

**Learning Rate (Step Size)** - The fraction value you multiply the gradient by to determine a new value in your Gradient Decent
* if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.

**Goldylocks** - the term given to the "perfect" learning rate for a given regression problem.
* Can be calculated by using 1/f(x)^ii (the inverse of the second derivative of f(x) at x) for a one-dimension problem.

**Batch** - the total number of examples you use to calculate the gradient in a single iteration.
* **Smaller** batches might not represent the whole set of data
* **larger** batches might have too much redundancy which costs more to compute without any extra payoff
* You must find a happy medium. Usually with random smaller sets and taking the average.

**Mini-batch stochastic gradient descent (mini-batch SGD)** - is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

# 2020-07-09
## First Steps with TF
### Python libraries
* NumPy
* Pandas

How to create new columns in dataframe:

```py
# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)
```
How to view only part of your dataframe
```py
print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])
```
It's important to note that DataFrames are assigned by reference, so if you want a copy you have to call the copy function.

### Linear Regression with tf.keras
**Hyper Parameter Tuning** is the process of modifying the Learning Rate, Epochs, and Batch Size to optimize your learning time.
* **Learning Rate** - A scalar used to train a model via gradient descent. During each iteration, the gradient descent algorithm multiplies the learning rate by the gradient. The resulting product is called the gradient step.
* **Epochs** - A full training pass over the entire dataset such that each example has been seen once. Thus, an epoch represents N/batch size training iterations, where N is the total number of examples.
* **Batch Size** - The number of examples you look at before recalculating your weights. One epoch spans sufficient iterations to process every example in the dataset. For example, if the batch size is 12, then each epoch lasts one iteration. However, if the batch size is 6, then each epoch consumes two iterations.
* **Convergence** - Informally, often refers to a state reached during training in which training loss and validation loss change very little or not at all with each iteration after a certain number of iterations. In other words, a model reaches convergence when additional training on the current data will not improve the model.

**Correlation Matrix** - indicates how each attribute's raw values relate to the other attributes' raw values. Correlation values have the following meanings:
* 1.0: perfect positive correlation; that is, when one attribute rises, the other attribute rises.
* -1.0: perfect negative correlation; that is, when one attribute rises, the other attribute falls.
* 0.0: no correlation; the two column's are not linearly related.

In general, the higher the absolute value of a correlation value, the greater its predictive power. For example, a correlation value of -0.8 implies far more predictive power than a correlation of -0.2.


# 2020-07-20
## Generalization
**Overfit Model** - When a training model becomes too complex and to specific to the training set that it is no longer effective at generalizing new data that has not been seen before.
**Ockham's Razor** - The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.
**Generalization Bounds** - statistical description of a model's ability to generalize to new data based on factors such as:
* the complexity of the model
* the model's performance on training data

A machine learning model aims to make good predictions on new, previously unseen data. But if you are building a model from your data set, how would you get the previously unseen data? Well, one way is to divide your data set into two subsets:
* **training set** - a subset to train a model.
* **test set** - a subset to test the model.

The following three basic assumptions guide generalization:
* We draw examples independently and identically (i.i.d) at random from the distribution. In other words, examples don't influence each other. (An alternate explanation: i.i.d. is a way of referring to the randomness of variables.)
* The distribution is stationary; that is the distribution doesn't change within the data set.
* We draw examples from partitions from the same distribution.

**Overfit Model** - When a training model becomes too complex and to specific to the training set that it is no longer effective at generalizing new data that has not been seen before.

## Training and Test Sets
When splitting your data into test and training sets try to make sure that your test set meets the following two conditions:
* Is large enough to yield statistically meaningful results.
* Is representative of the data set as a whole. In other words, don't pick a test set with different characteristics than the training set.

**NEVER TRAIN ON TEST DATA!** If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. For example, high accuracy might indicate that test data has leaked into the training set.

When we have massive pools of data to pull from, we could get away with a 80:20 split of training:test. However if our data is relatively smart we might have to do other things like cross validation.

## Validation Set
Instead of just splitting your data into two partitions, test and training. There is one more partition that we should make. The validation set.

This allows us to:
1. Train on the training set
2. validate the training using the validation set
3. tweak our model (change hyper parameters, add/remove features, or even restart entirely) based on how well we fit the validation set
4. Then finally check that we have not over-fit to the validation set by testing our model against the test set.
   * We should see results as close to our results on the validation set as possible.

# 2020-07-22
## Representation
### Feature engineering
The process of transfroming raw data into a feature vector. This takes up a significant portion of our time in machine learning. Most machine learning models must represent features as numbers. So when we run into features that are strings or other types of data, we will need a way to transform them.

**Categorical Features** - Features that have a discrete set of possible values. Street name, for example. These values will need to be converted to numberic values. We do this by putting all of the possible feature values in an enumerated mapping and storing only the index as the feature's value.

**OOV (out-of-vocabulary) Bucket** - When dealing with categorical features, it is likely that our dataset does not contain every possible value for a feature, so we put all of those extra values into a catch-all "other" category known as an OOV bucket.

For example using this approach, here's how we can map our street names to numbers:
* map Charleston Road to 0
* map North Shoreline Boulevard to 1
* map Shorebird Way to 2
* map Rengstorff Avenue to 3
* map everything else (OOV) to 4

However this solution can create some issues.
* We will be learning a single weight that applies to all streets and multiplying that weight based upon the location of the street in the map (0, 1, 2, etc). Our model will need the flexibility of learning different weights for each street.
* We might have cases where a house is on a courner of two streets wich means that street\_name would take multiple values

**One-hot-encoding** - When you take categorical features and convert them into a binary vector where the length of the vector is the size of the discrete set of possible values. All values of the vector are set to 0 except for the one that represents the value of the feature.

**Multi-hot-encoding** - Same as one-hot-encoding except there can be multiple 1 values in the vector.

**Sparse Representation** - Suppose that you had 1,000,000 different street names in your data set that you wanted to include as values for street_name. Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a very inefficient representation in terms of both storage and computation time when processing these vectors. In this situation, a common approach is to use a sparse representation in which only nonzero values are stored.

### Qualities of Good Features
##### Avoid rarely used or discrete feature values
Good features should have values that appear more than 5 or so times in a data set. This allows the model to see the feature and it's relation with other features and determine if it is a good predictor of the label. Conversely if a feature's value appears only once or very rarely, the model can't make predictions based on that feature.
* A unique id, for example, would be a bad feature because each value would only be used once.
##### Prefer clear and obvious meanings
Each feature should have clear meaning
* For example use years instead of seconds to denote the age of a house.
##### Don't use "magic" values
Using the value -1 to mean that a value isn't present is an example of a "magic" value that has some extra meaning. Instead we should add an additional boolean feature to denote whether the value exists or not.
##### Account for upstream instability
Definitions of features should not change over time.
* For example, using cityName instead of cityId because CityName is much less likely to change than the Name.

### Cleaning Data
**Scaling** - When we convert floating point feature values from their natural range into a standard range. (Ex: 1 to 1000 -> 0 to 1, or -1 to +1)
* This helps with gradient descent convergence
* Helps avoid the "NaN trap" when a value in the model exceeds the floating-point precision limit during training from being multiplied too many times.
* Helps the model learn apprpriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having larger ranges.
   * Nothing terrible will happen if Feature A is scaled from -1 to +1 while Feature B is scaled from -3 to +3. However, your model will react poorly if Feature B is scaled from 5000 to 100000.
##### Handling Outliers in Data
**Logarithmic Scaling** - When you take the log of each value.

**Capping Feature Values** - set a maximum or minimum value for feature values to stop outliers from having excessive influence.

**Binning** - When you split a feature with a large range of values that don't exactly have a linear relationship with your label. Take Lattitude for example, each different lattitude has an effect on housing price, however latitude of 34 is not proportionately lower or higher than latitude 36. We split these lattitudes into 'zones' in boolean vector similar to hot-encoding so the model can learn completely different weights for each zone.
##### Scrubbing
Until now, we've assumed that all the data used for training and testing was trustworthy. In real-life, many examples in data sets are unreliable due to one or more of the following:
* Omitted values. For instance, a person forgot to enter a value for a house's age.
* Duplicate examples. For example, a server mistakenly uploaded the same logs twice.
* Bad labels. For instance, a person mislabeled a picture of an oak tree as a maple.
* Bad feature values. For example, someone typed in an extra digit, or a thermometer was left out in the sun.

In addition, getting statistics like the following can help:
* Maximum and minimum
* Mean and median
* Standard deviation
* Consider generating lists of the most common values for discrete features. For example, do the number of examples with country:uk match the number you expect. Should language:jp really be the most common language in your data set

Follow these rules:
* Keep in mind what you think your data should look like.
* Verify that the data meets these expectations (or that you can explain why it doesn't).
* Double-check that the training data agrees with other sources (for example, dashboards).

## Feature Crosses
**Feature Cross** - a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term cross comes from cross product.)

There are many different kinds of Feature Crosses:
* [A X B]: a feature cross formed by multiplying the values of two features.
* [A x B x C x D x E]: a feature cross formed by multiplying the values of five features.
* [A x A]: a feature cross formed by squaring a single feature.

#### Crossing One-Hot Vectors
Suppose we have two features: country and language. A one-hot encoding of each generates vectors with binary features that can be interpreted as country=USA, country=France or language=English, language=Spanish. Then, if you do a feature cross of these one-hot encodings, you get binary features that can be interpreted as logical conjunctions, such as:

`country:usa AND language:spanish`

Two 5 element one-hot vectors crossed would give you one 25 element one-hot vector for all the possible combinations of the two one-hot vector values.

**By doing this we end up with vastly more predictive ability than either feature on its own.** For example, if a dog cries (happily) at 5:00 pm when the owner returns from work will likely be a great positive predictor of owner satisfaction. Crying (miserably, perhaps) at 3:00 am when the owner was sleeping soundly will likely be a strong negative predictor of owner satisfaction.

## Regularization: Simplicity
Often times when we are training data we run into the problem of overfitting. In order to solve this problem we instead of simply aiming to minimize loss (empirical risk minimization):

`minimize(Loss(Data|Model))`

we'll now minimize loss+complexity (**Structural Risk Minimization**)

`minimize(Loss(Data|Model) + complexity(Model))`

Our training optimization algorithm is now a function of two terms: the loss term, which measures how well the model fits the data, and the regularization term, which measures model complexity.

We have two approaches to defining **model complexity**:
* function of all the weights of all of the features in the model
* function of the total number of features with nonzero weights

**L2 regularization** - formula, which defines the regularization term as the sum of the squares of all the feature weights. In this formula, <ins>weights close to zero have little effect on model complexity</ins>, while outlier weights can have a huge impact.

### Lambda
Model developers tune the overall impact of the regularization term by multiplying its value by a scalar known as lambda (also called the regularization rate). That is, model developers aim to do the following:

`minimize(Loss(Data|Model) +  λ*complexity(Model))`

Performing L2 regularization has the following effect on a model

* Encourages weight values toward 0 (but not exactly 0)
* Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution.

Increasing the lambda value strengthens the regularization effect. For example, the histogram of weights for a high value of lambda might look as shown in Figure 2.

When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:
* **If your lambda value is too high** - your model will be simple, but you run the risk of underfitting your data. Your model won't learn enough about the training data to make useful predictions.
* **If your lambda value is too low** - your model will be more complex, and you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.

