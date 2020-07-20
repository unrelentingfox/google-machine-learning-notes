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

