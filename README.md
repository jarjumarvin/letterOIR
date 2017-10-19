# Neural Network for MNIST Handwritten Digit Classification

Implementation of a multi-layer neural network based on the example from Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) and [karandesai-96](http://github.com/karandesai-96)'s implementation.

### Dependencies

- numpy
- wget
- matplotlib

### Usage

- Initializing (784 input neurons, as 28x28 = 784 pixels, 30 hidden neurons and 10 output neurons for each of the digits)
```
import neuralnet
---
net = neuralnet.Network([784, 30, 10], iterations=10)
```
- Retrieving MNIST data and training the network
```
import utils
---
data = utils.load_mnist()

training_data = data[0]
validation_data = data[1]
testing_data = data[2]

net.train(training_data)
```
- Test the network for it's accuracy
```
print('Accuracy is {0}% on a sample size of {1} test samples.'.format(net.validate(data[1]), len(data[1])))
--> e.g. "Accuracy is 93.41% on a sample size of 10000 test samples."
```
- Saving and loading the network parameters to/from a file
```
net.save('mymodel.npz')
----
net = neuralnet()
net.load('mymodel.npz')
```
- Plot a random image and the networks prediction for it
```
x = utils.get_random_image(data[2])
utils.plot_prediction(x, net.predict(x))
```
<p align="center">
  <img src="https://i.imgur.com/umZ6h2T.png"/>
</p>
