# 2020-07-09
## Framing Key ML Terminology

**Label** - (y) is the thing we are predicting.

**feature** - (x) is an input variable that is used to predict the label.

**Training** - means creating or learning the model.

**Inference** - meany applying the trained model to the unlabeled examples.

**Regression Model** - predicts the continuous values
* What is the value of a house?
* What is the probability that a user will click an add?

**Classification Model** - predicts discrete values
* is an email spam?
* is this image a dog or a cat or a hamster?

## Decending into ML
A simple 2d model can be defined with: y1 = b + w1 * x1
* y is the target value
* w is the weight values
* x is the input
* b is the bias. (the y intercept, sometimes referred to as w0)

Some models might depend on more than one input making the equation something like: y1 = b + w1 * x1 + w2 * x2 ..

**Loss** - is how close the predictions are to the model.

**L2 Loss (Squared Loss)** - square of the difference between prediction and label (y - y1)^2
* this aplifies the loss value exponentially as the distance from the prediction increases.

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




