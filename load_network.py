from network import Network
from layer import ParallelLayer, SoftplusLayer, TanhLayer, SoftmaxLayer
import numpy as np
import os.path

def get_name(file):
    base_path   = os.path.dirname(os.path.realpath(__file__))
    local_file  = base_path + os.path.normpath('/weights/'+file)
    nested_file = base_path + os.path.normpath('/queens_speech/weights/'+file)
   
    if os.path.isfile(local_file):
        return local_file
    elif os.path.isfile(nested_file):
        return nested_file
    else:
        raise Exception("Weights file could not be found")

def load_network():
    weights_0 = np.loadtxt(get_name('0_weights.arr'))
    weights_1 = np.loadtxt(get_name('1_weights.arr'))
    weights_2 = np.loadtxt(get_name('2_weights.arr'))
    weights_3 = np.loadtxt(get_name('3_weights.arr'))

    bias_0 = np.loadtxt(get_name('0_bias.arr'))
    bias_1 = np.loadtxt(get_name('1_bias.arr'))
    bias_2 = np.loadtxt(get_name('2_bias.arr'))
    bias_3 = np.loadtxt(get_name('3_bias.arr'))

    net = Network([
        ParallelLayer(SoftplusLayer(250,50),3),
        TanhLayer(150,500),
        TanhLayer(500,500),
        SoftmaxLayer(500,250)
    ])

    net.layers[0].layer.weights = weights_0
    net.layers[0].layer.bias = bias_0

    net.layers[1].weights = weights_1
    net.layers[1].bias = bias_1

    net.layers[2].weights = weights_2
    net.layers[2].bias = bias_2

    net.layers[3].weights = weights_3
    net.layers[3].bias = bias_3

    return net