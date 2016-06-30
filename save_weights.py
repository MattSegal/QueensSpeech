import cPickle
import numpy as np
import os.path

def get_name(file):
    base_path = os.path.dirname(os.path.realpath(__file__))
    return base_path + os.path.normpath('/weights/'+file)

f = open('net.pkl','rb')

net = cPickle.loads(f.read())

parallel_layer = net.layers[0]
hidden_layer_1 = net.layers[1]
hidden_layer_2 = net.layers[2]
softmax_layer = net.layers[3]

np.savetxt(get_name('0_weights.arr'),parallel_layer.layer.weights)
np.savetxt(get_name('1_weights.arr'),hidden_layer_1.weights)
np.savetxt(get_name('2_weights.arr'),hidden_layer_2.weights)
np.savetxt(get_name('3_weights.arr'),softmax_layer.weights)

np.savetxt(get_name('0_bias.arr'),parallel_layer.layer.bias)
np.savetxt(get_name('1_bias.arr'),hidden_layer_1.bias)
np.savetxt(get_name('2_bias.arr'),hidden_layer_2.bias)
np.savetxt(get_name('3_bias.arr'),softmax_layer.bias)
