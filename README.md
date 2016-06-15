# NeuralNet
Low dependency feed-forward neural network with back propagation written in python

## Usage
```python
import Network from NeuralNetwork
network = Network( 2, 1, [3,4] ) # inputs, outputs, hidden layer makeup (ie: 2 layers, first with 3 neurons, the second with 4)
outputs = network.getOutput( inputs ) # takes a list (must be equal in size to inputs) of input values,
      # returns a list (equal in size to outputs) of the resulting output values
network.adjustWeights( targetOutputs ) # This handles the back-propagation.
      # MUST be called after getOutput. 
      # Takes a list (must be equal in size to outputs) of the TARGET outputs, 
      # adjusts the network based on the network's learning rate
```
