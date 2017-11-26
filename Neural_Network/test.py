# -*- coding: utf-8 -*-

from Neural_Network import nn

import numpy as np

input_data = np.ones((100, 1))

net = nn.FullyConnectedLayer(input_data, 10, 'sigmoid')

output_data = net.forward()

print(output_data)