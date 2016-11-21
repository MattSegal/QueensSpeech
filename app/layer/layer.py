import numpy as np

class Layer(object):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Default hyperparameters
        initial_weights = 0.01
        self.learning_rate = 0.01

        self.weights = np.zeros((num_inputs,num_outputs))
        self.bias = np.zeros(num_outputs)
        self.initialize_weights(initial_weights)

        # Dimensions depend on batch size
        self.output = np.array([]) 
        self.input = np.array([])

        # Momentum method variables
        self.weight_deriv = np.zeros((num_inputs, num_outputs))
        self.bias_deriv = np.zeros(num_outputs)
        self.momentum = 0.8

    def set_learning_rate(self,rate):
        self.learning_rate = rate

    def set_momentum(self,momentum):
        self.momentum = momentum

    def initialize_weights(self, init_factor):
        """
        Set pre-learning weights to random numbers between +/- init_factor
        """
        self.weights = 2 * init_factor * np.random.rand(self.weights.shape[0], self.weights.shape[1]) - init_factor
        self.bias    = 2 * init_factor * np.random.rand(self.bias.shape[0]) - init_factor

    def get_output(self, inputs):
        """
        Gets the output value of this layer given the inputs
        inputs is array of shape (batch_size, num_inputs)
        output is array of shape (batch_size, num_outputs)
        """
        assert inputs.shape[1] == self.weights.shape[0]
        self.input = inputs
        self.linear_output = inputs.dot(self.weights) + self.bias
        self.output = self.activation(self.linear_output)
        return self.output

    def get_error(self, target):
        """
        Calculates squared error cost of output state given a target set
        This is the default cost function, you can override this and get_error_deriv.
        """
        assert self.output.shape == target.shape
        return np.sum(0.5 * np.square(target - self.output))

    def get_error_deriv(self, target):
        """
        Calculates squared error error derivative of output layer given a target set
        This assumes that this layer is the final layer in the network.
        """
        assert self.output.shape == target.shape
        return -1*(target - self.output)

    def next_error_deriv(self, error_deriv):
        """
        Take the derivative of the error with respect to this layer's output,
        and return the derivative of the error with respect to the layer's inputs.
        
        This is used to transfer error information to downstream layers in backpropagation.
        """
        return np.dot(self.activation_deriv() * error_deriv, self.weights.T)

    def update_weights(self, error_deriv):
        """
        Calculate the weights from the layer's output error derivative.
        error_deriv (batch x num_out)
            change in error for change in layer output

        weight_deriv (num_in x num_out)
            change in error for change in each weight
        """
        assert self.input.size > 0
        batch_size = float(self.input.shape[0])

        # has same shape as output
        error_term = error_deriv * self.activation_deriv()

        self.weight_deriv = self.input.T.dot(error_term) / batch_size + self.weight_deriv * self.momentum
        delta_weights = - self.learning_rate * self.weight_deriv
        self.weights += delta_weights

        self.bias_deriv = np.sum(error_term, 0) / batch_size + self.bias_deriv * self.momentum
        delta_bias = - self.learning_rate * self.bias_deriv
        self.bias += delta_bias

    def activation(self, z):
        """
        Function applied to the linear sum of inputs and weights.
        This gives the layer its non-linear characteristics.
        """
        raise NotImplementedError()

    def activation_deriv(self):
        """
        Derivative of activation function.
        This is used in backpropagation.
        """
        raise NotImplementedError()

