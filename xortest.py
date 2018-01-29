#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 08:55:09 2017

@author: harshavardhan
"""

import extract as ext
import neural_net as nn
import csv
import numpy as np
import sys
from sklearn import preprocessing as pp

xtrain = [[0,0],[0,1],[1,0],[1,1]]
ytrain = [[1,0],[0,1],[0,1],[1,0]]
ytrain_label = [0,1,1,0]
n_hidden = [4]
num_classes = 2
Neural_net = nn.neural_network(2,n_hidden,num_classes)
Neural_net.train(xtrain, ytrain, ytrain_label, 0.01,10)
