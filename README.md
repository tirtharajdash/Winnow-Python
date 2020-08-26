# Winnow-Python

Python Implementation of Nick Littlestone's Winnow Algorithm

### Paper

Nick Littlestone had proposed a simple machine learning technique for learning linear classifier from labeled instances (i.e. supervised learning). Winnow is very similar to Perceptron- a simple single layered neural network. However, the perceptron adopts additive weight-update scheme (sometimes, with gradient descent); whereas Winnow uses a multiplicative weight update scheme. 

Winnow algorithm is used for settings where there are a possibly infinitely many boolean (or real) attributes and many of those dimensions are irrelevant. The name comes from this fact. It can scale to high-dimensional data easily and learning is fast. It was initially proposed for Online Learning settings. 
`
Nick Littlestone (1988). "Learning Quickly When Irrelevant Attributes Abound: A New Linear-threshold Algorithm", Machine Learning 285–318(2).
`

### Implementation

The implementation in this repository is little different from Winnow in the following ways:

 - I have added a epoch-based learning
 - The version of Winnow has a demotion step where weights are not made 0, rather divided by the multiplicative factor (\alpha). This is called "Winnow2".
 - I have implemented early stopping with a validation set to make the model capable of generalisation.

### Cite as

Tirtharaj Dash (2020). Winnow-Python (https://github.com/tirtharajdash/Winnow-Python), GitHub. Retrieved Month DD, YYYY.
