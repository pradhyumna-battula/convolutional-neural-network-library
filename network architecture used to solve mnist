import convolve as nn
import numpy as np

net = nn.Network([
    nn.ConvolutionalLayer((28,28,1), 5, 16),      # → (24,24,16)
    nn.LeakyReluLayer(),
    nn.ConvolutionalLayer((24,24,16), 5, 16),     # → (20,20,16)
    nn.LeakyReluLayer(),
    nn.ConvolutionalLayer((20,20,16), 9, 8),      # → (12,12,8)
    nn.LeakyReluLayer(),
    nn.FlattenLayer(),                            # → 12*12*8 = 1152
    nn.DenseLayer(1152, 128),
    nn.LeakyReluLayer(),
    nn.DenseLayer(128,10),
    nn.SoftmaxEntropyLayer()
], nn.SparseCategoricalCrossEntropy())

