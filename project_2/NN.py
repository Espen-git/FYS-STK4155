import functions
import activation_functions as af
import numpy as np
#from numpy.core.defchararray import not_equal
#from numpy.core.fromnumeric import argmax, argmin
#import activation as act

np.random.seed(1337)

class Layer():
    def __init__(self, num_inputs, num_outputs, activation_function):
        """
        inputs - Number of inputs to the layer (number of nodes in previous layer)
        outputs - Number of outputs (nodes) of this layer
        """

        self.num_in = num_inputs
        self.num_out = num_outputs
        self.af = activation_function
        # initializing weights form a normal distribution
        self.weights = np.random.randn(num_inputs, num_outputs)
        
        if self.af.name == "Sigmoid":
            self.weights = self.weights * np.sqrt(1.0 / num_outputs)
        elif self.af.name == "ReLu" or self.af.name == "LeakyReLU":
            self.weights = self.weights * np.sqrt(2.0 / num_inputs)

        self.bias = 0.001 * np.ones((1, num_outputs))

    def __call__(self, X):
        self.z = (X @ self.weights) + self.bias
        self.out = self.af(self.z)
        self.dout = self.af.derivative(self.z) # Derivative of output
        return self.out

class NN():
    def __init__(self, num_inputs, num_outputs, activation_function,
                 activation_function_output, cost_function,
                 num_nodes_hidden_layers=None):

        self.num_in = num_inputs
        self.num_out = num_outputs
        self.af = activation_function
        self.af_out = activation_function_output
        self.cf = cost_function
        self.num_nodes_each_h = num_nodes_hidden_layers
        self.num_h_layers = len(self.num_nodes_each_h)

        self.make_layers()

    def make_layers(self):
        self.layers = []

        if self.num_nodes_each_h:
            first_layer = Layer(self.num_in, self.num_nodes_each_h[0],
                          self.af)
            self.layers.append(first_layer)

            # Now the hidden layers
            for i in range(self.num_h_layers - 1):
                hidden_layer = Layer(self.num_nodes_each_h[i],
                                     self.num_nodes_each_h[i+1],
                                     self.af)

                self.layers.append(hidden_layer)
            
            output_layer = Layer(self.num_nodes_each_h[-1], self.num_out,
                                 self.af_out)
            self.layers.append(output_layer)

        else: # No hidden layers
            layer = Layer(self.num_in, self.num_out, self.af_out)
            self.layers.append(layer)

        return self.layers
 
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)

        return X

    def back_prop(self, X, z, eta=0.001, lmd=0):
        self.forward(X) # generate self.out for layers objects

        dcf = self.cf.derivative(self.layers[-1].out, z)
        # output error
        oe = dcf * self.layers[-1].dout

        # Update weights and bias at last layer / output
        self.layers[-1].weights = self.layers[-1].weights - eta * (self.layers[-2].out.T @ oe)
        self.layers[-1].bais = self.layers[-1].bias - eta * oe[0,:]

        # Uppdate weights and bias at hidden layers
        for i in reversed(range(1, len(self.layers)-1)):
            oe = (oe @ self.layers[i + 1].weights.T) * self.layers[i].dout
            self.layers[i].weights = self.layers[i].weights - eta * (self.layers[i - 1].out.T @ oe) - 2 * eta * lmd * self.layers[i].weights 
            self.layers[i].bias = self.layers[i].bias - eta * oe[0,:]

        # Uppdate weights and bias at first hidden layer (from input)
        oe = (oe @ self.layers[1].weights.T) * self.layers[0].dout
        self.layers[0].weights = self.layers[0].weights - eta * (X.T @ oe) - 2 * eta * lmd * self.layers[0].weights
        self.layers[0].bias = self.layers[0].bias - eta * oe[0,:]

    def logistic(self, X, z, eta=0.001, lmd=0.1):
        self.forward(X)

        dcf = self.cf.derivative(self.layers[-1].out, z)
        oe = dcf * self.layers[0].dout

        self.layers[0].weights = self.layers[0].weights - eta * (X.T @ oe) - 2 * eta * lmd * self.layers[0].weights
        self.layers[0].bias = self.layer[0].bias - eta * oe[0,:]