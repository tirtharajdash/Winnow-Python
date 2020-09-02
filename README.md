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

 - I have added a epoch-based learning (this would mean that this may not be considered for online-learning-- single pass)
 - The version of Winnow has a demotion step where weights are not made 0, rather divided by the multiplicative factor (\alpha). This is called "Winnow2".
 - I have implemented early stopping with a validation set to make the model capable of generalisation.
 - The parameters of Winnow2 (\alpha and threshold) are tuned.


I will keep on adding new variants of this as and when ready. Thanks for watching!


Note that the notebook file (.ipynb) and .py file may be different. The latest is the .py file.


### Sample outputs

$ python winnow2.py (verbose was set to False)\
Dimension of Data ( Instances:  2434 , Features:  5000  )\
Dimension of X:  (2434, 5000)\
Dimension of y:  (2434,)\

Dimension of X_train:  (1947, 5000)\
Dimension of y_train:  (1947,)\
Dimension of X_val:  (487, 5000)\
Dimension of y_val:  (487,)\

Dimension of Data_test ( Instances:  1044 , Features:  5000  )\
Dimension of X:  (1044, 5000)\
Dimension of y:  (1044,)\
-----------------------------------------------------------------\
Trying::	 alpha: 2 threshold: 5000 	( 1 of 6 )\
-----------------------------------------------------------------\
[Selected] at epoch 0 	train_perf: 0.6292 	val_perf: 0.6304\
[Selected] at epoch 2 	train_perf: 0.6584 	val_perf: 0.6550\
[Selected] at epoch 3 	train_perf: 0.6826 	val_perf: 0.6715\
[Selected] at epoch 6 	train_perf: 0.6785 	val_perf: 0.6735\
[Selected] at epoch 13 	train_perf: 0.7211 	val_perf: 0.6838\
[Selected] at epoch 17 	train_perf: 0.7278 	val_perf: 0.6879\
[Selected] at epoch 36 	train_perf: 0.7468 	val_perf: 0.6920\
[Selected] at epoch 55 	train_perf: 0.7519 	val_perf: 0.6940\
[Selected] at epoch 62 	train_perf: 0.7458 	val_perf: 0.7002\
Stopping early after epoch 83\
Model found and saved.

-----------------------------------------------------------------\
Trying::	 alpha: 2 threshold: 2500 	( 2 of 6 )\
-----------------------------------------------------------------\
Stopping early after epoch 20\
No better model found.\

-----------------------------------------------------------------\
Trying::	 alpha: 3 threshold: 5000 	( 3 of 6 )\
-----------------------------------------------------------------\
...\


### Cite as

Tirtharaj Dash (2020). Winnow-Python (https://github.com/tirtharajdash/Winnow-Python), GitHub. Retrieved Month DD, YYYY.
