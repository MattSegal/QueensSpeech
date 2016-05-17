from layer import Layer
import numpy as np

class LogisticLayer(Layer):
    """
    If you use this guy as your final layer for classification,
    then your training output values should take the range of approximately [0,1]
    """
    def activation(self, z):
        """
        Sigmoid function. 
        Output can be interpreted as a probability.
        z is an array.
        """
        return 1.0 / (1.0 + np.exp(-z))

    def activation_deriv(self):
        """
        Derivative of Sigmoid function w.r.t inputs.
        """
        assert self.output.size > 0
        return np.multiply(self.output, (1 - self.output))
