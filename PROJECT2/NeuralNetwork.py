import numpy as np


# Activation functions and derivatives

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(Z):
    sig = sigmoid(Z)
    return  sig * (1 - sig)

def relu(Z):
    return np.maximum(0,Z)

def relu_backward(Z):
    Z = Z.copy()
    Z[Z > 0] = 1
    Z[Z <= 0] = 0
    return Z

# Cost functions and derivatives

def mse_loss(pred, actual):
    assert pred.shape == actual.shape, "Shape mismatch"
    return 1/2.0 * np.mean(np.square(pred - actual))

def mse_backward(pred, actual):
    assert pred.shape == actual.shape, "Shape mismatch"
    return (pred - actual)

def logistic_loss(pred, actual):
    assert pred.shape == actual.shape, f"Shape mismatch. Pred shape : {pred.shape}. Actual shape : {actual.shape}"
    return (-actual * np.log(pred) - (1 - actual) * np.log(1 - pred)).mean()

def logistic_backward(pred ,actual):
    assert pred.shape == actual.shape, "Shape mismatch"
    Y_hat, Y = pred, actual
    return - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))



ACTIVATIONS = {'sigmoid': sigmoid,
                'relu':relu,
                'linear': lambda x : x
                }

D_ACTIVATIONS = {'sigmoid': sigmoid_backward,
                'relu':relu_backward,
                'linear' : lambda x: 1
                }

COSTS = {'mse': mse_loss,
        'logistic' : logistic_loss}

D_COSTS = {'mse': mse_backward,
            'logistic' : logistic_backward}


class Input():
    def __init__(self, input_shape):
        """
        input_shpe : int
            feature dimension of X
        """
        self.input_shape = input_shape
        self.nodes = input_shape
        
        
        self.type = 'input'


    def connect(self, in_connection, out_connection):
        self.in_connection = None
        self.out_connection = out_connection


class Layer():
    def __init__(self, nodes, activation):
        self.nodes = nodes
        self.activation = activation
        self.type = 'linear'

    def connect(self, in_connection, out_connection):
        """
        Connect the nodes together to form a graph to 
        traverse for forward and backward propagation
        """

        self.in_nodes = in_connection.nodes

        if (in_connection.type == 'input') & (out_connection is None):
            self.in_connection = None
            self.out_connection = None
            self.W = np.random.randn(self.nodes, self.in_nodes) * 0.1
            self.b = np.random.randn(self.nodes, 1) * 0.1
        else:
            if out_connection:
                self.out_nodes = out_connection.nodes

            self.in_connection = in_connection
            if in_connection.type == 'input':
                self.in_connection = None
            
            self.out_connection = out_connection
            
            self.W = np.random.randn(self.nodes, self.in_nodes) * 0.1
            self.b = np.random.randn(self.nodes, 1) * 0.1
        
    def forward(self, X):
        if self.in_connection is None:
            Z = np.dot(self.W, X.T) + self.b
        else:
            Z = np.dot(self.W, self.in_connection.A) + self.b

        A = ACTIVATIONS[self.activation](Z)
        
        self.Z = Z
        self.A = A

    def backward(self, X, y, cost_function):

        # Layers not including output layer
        if self.out_connection is not None:
            self.delta = np.dot(self.out_connection.W.T, self.out_connection.delta) * D_ACTIVATIONS[self.activation](self.Z)
            if self.in_connection is not None:
                self.dW = self.delta.dot(self.in_connection.A.T)
            else:
                self.dW = self.delta.dot(X)
            
            self.dW = self.dW / X.shape[0]
            self.db = self.delta.sum(axis = 1, keepdims = True) / X.shape[0]
        
        # Output layer
        else:
            # print(self.A.shape, y.shape, self.Z.shape)
            self.delta = D_COSTS[cost_function](self.A, y.T) * D_ACTIVATIONS[self.activation](self.Z)
            if self.in_connection:
                self.dW = self.delta.dot(self.in_connection.A.T)
            else:
                self.dW = self.delta.dot(X)
            # self.dW = self.in_connection.A.dot(self.delta.T).T 
            self.dW = self.dW / X.shape[0]
            self.db = self.delta.mean(axis = 1).reshape(self.nodes, 1)


class NeuralNetwork():
    def __init__(self, X, y, validation_data, nodes, activations, cost_function):
        """
        X : numpy array
            shape : (num_examples, num_features)
        y : labels

        nodes : list
            list of the number of nodes in each layer

        activations : list
            list with the activation function for each layer

        cost_function : string
            name of the cost function used
        

        example:

            # Training a network with two hidden layer of size 10 and 1-dimensional output layer.
            nn = NeuralNetwork(X, y, [10,10,1], ['sigmoid', 'sigmoid', 'linear'], "mse")

        """

        assert len(y.shape) > 1, f"y has shape {y.shape}. Need more dimensions."
        self.X = X
        self.y = y

        self.validation_data = validation_data
        self.nodes = nodes
        self.activations = activations
        self.cost_function = cost_function

        assert len(nodes) == len(activations), "nodes and activations must have same lenght."

        self.input = Input(X.shape[1])
        self.layers = [Layer(node, activation) for node, activation in zip(nodes, activations)]
        self._connect_layers()

    def _connect_layers(self):
        if len(self.layers) == 1:
            for layer in self.layers:
                layer = layer.connect(self.input, None)
        else:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    layer = layer.connect(self.input, self.layers[i+1])
                elif i < len(self.layers)-1:
                    layer = layer.connect(self.layers[i-1], self.layers[i+1])
                else:
                    layer =  layer.connect(self.layers[i-1], None)

    def train(self, batch_size, epochs, learning_rate, l2_lambda = 0, keep_history = False):
        N = len(self.X)
        
        iterations = int(np.ceil(N / batch_size))

        self.history = []

        for epoch in range(epochs):
            for i in range(iterations):

                X = self.X[i * batch_size : (1+i)*batch_size]
                y = self.y[i * batch_size : (1+i)*batch_size]
                
                # Forward pass
                for layer in self.layers:
                    layer.forward(X)
                
                # Backward pass
                for layer in reversed(self.layers):
                    layer.backward(X, y, self.cost_function)
                
                # Weight update
                for layer in self.layers:
                    layer.W = layer.W - learning_rate * layer.dW - learning_rate * l2_lambda * layer.W / X.shape[0]
                    layer.b -= learning_rate * layer.db

            self.history.append([epoch, 
            COSTS[self.cost_function](self.predict(self.validation_data[0]), self.validation_data[1]),
            COSTS[self.cost_function](self.predict(X), y)])

            def print_results():
                print(f"EPOCH {epoch} :")
                print(f"{self.cost_function} cost:", COSTS[self.cost_function](self.predict(X), y))
                val_pred = self.predict(self.validation_data[0])
                print(f"validaltion {self.cost_function} cost:", COSTS[self.cost_function](val_pred, self.validation_data[1]))
                print("\n")
                    
            if (epochs > 10):
                if (epoch%(epochs//10) == 0):
                    print_results()
            else:
                print_results()

        print_results()

    def predict(self, X):
        for layer in self.layers:
            layer.forward(X)
        return self.layers[-1].A.T


        
