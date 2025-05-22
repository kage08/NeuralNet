#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:29:59 2017

@author: harshavardhan
"""

import random
import numpy
from math import exp
from matplotlib import pyplot as plt
from sklearn import metrics as mt
import copy
from sklearn.utils import shuffle

class neural_network:
    def __init__(self, input_layer_n, n_hidden_layers_counts, num_classes):
        random.seed(1)
        numpy.random.seed(1)
        self.input_layer_n = input_layer_n
        self.n_hidden_layers_counts = n_hidden_layers_counts
        self.num_classes = num_classes
        
        self.hidden_layers = []
        prev_num_neurons = input_layer_n
        
        # Initializing weights for hidden layers
        for num_neurons in self.n_hidden_layers_counts:
            layer = {
                'weights': (numpy.random.rand(num_neurons, prev_num_neurons + 1) / 100.0),
                'derivatives': numpy.zeros(num_neurons),
                'outputs': numpy.zeros(num_neurons),
                'outputs_with_bias': numpy.zeros(num_neurons + 1) # For storing outputs concatenated with 1
            }
            self.hidden_layers.append(layer)
            prev_num_neurons = num_neurons
            
        # Initializing weights for output layer
        # The number of inputs to the output layer is prev_num_neurons (neurons in last hidden layer)
        self.output_layer = {
            'weights': (numpy.random.rand(num_classes, prev_num_neurons + 1) / 100.0),
            'derivatives': numpy.zeros(num_classes),
            'outputs': numpy.zeros(num_classes)
            # No 'outputs_with_bias' needed for the output layer itself in this structure
        }
        # To store the input to the network, potentially with bias, for backpropagation
        self.input_layer_outputs_with_bias = numpy.zeros(input_layer_n + 1)

    def forward_propogate(self, x):
       # Ensure x is a numpy array
       inputs = numpy.array(x) if not isinstance(x, numpy.ndarray) else x
       self.input_layer_values = inputs # Store input layer values (raw, without bias)

       # Prepare the input layer's output with bias for the first hidden layer
       self.input_layer_outputs_with_bias = numpy.concatenate((inputs, [1]))
       current_inputs_with_bias = self.input_layer_outputs_with_bias
       
       for i, layer in enumerate(self.hidden_layers):
           activations = layer['weights'] @ current_inputs_with_bias
           layer_outputs = self.sigmoid(activations)
           layer['outputs'] = layer_outputs
           # Store this layer's output with bias for the next layer or output layer
           layer['outputs_with_bias'] = numpy.concatenate((layer_outputs, [1]))
           current_inputs_with_bias = layer['outputs_with_bias']

       # For the output layer, the input is the 'outputs_with_bias' from the last hidden layer
       # If there are no hidden layers, the input is 'self.input_layer_outputs_with_bias'
       final_input_to_output_layer_with_bias = current_inputs_with_bias # This holds the last layer's output_with_bias

       output_activations = self.output_layer['weights'] @ final_input_to_output_layer_with_bias
       final_outputs = self.softmax(output_activations)
       self.output_layer['outputs'] = final_outputs
       return final_outputs

    def sigmoid(self, x):
        x = numpy.array(x) if not isinstance(x, numpy.ndarray) else x
        # Prevent overflow for large negative x by clipping or using a more stable sigmoid
        # For simplicity, direct numpy.exp is used here as in original, but be mindful of potential overflow.
        return 1.0 / (1.0 + numpy.exp(-x))

    def softmax(self, inp):
        inp = numpy.array(inp) if not isinstance(inp, numpy.ndarray) else inp
        exp_inp = numpy.exp(inp - numpy.max(inp)) # Numerical stability
        return exp_inp / numpy.sum(exp_inp)

    def train(self, xtrain, ytrain,ytrain_label, eta,gamma, n_epoch,test_while_train=False,xtest=list(), ytest_label=list(),draw_plot = False):
        epoch_n=list()
        accurtrain=list()
        accurtest = list()
        
        maxlayers_hidden_weights = []
        maxlayers_output_weights = None
        # Convert xtrain and ytrain to numpy arrays once at the beginning if they are lists
        xtrain_np = numpy.array(xtrain) if not isinstance(xtrain, numpy.ndarray) else xtrain
        ytrain_np = numpy.array(ytrain) if not isinstance(ytrain, numpy.ndarray) else ytrain
        # ytrain_label is for accuracy calculation, can remain a list or be converted
        # xtest also needs to be numpy array if test_while_train is True
        xtest_np = numpy.array(xtest) if xtest and not isinstance(xtest, numpy.ndarray) else xtest


        maxaccur = self.test(xtrain_np, ytrain_label) # Initial test with numpy arrays
        for i in range(n_epoch):
            print('Epoch No:',i)
            # Shuffle numpy arrays; ensure ytrain_label is shuffled consistently if it's separate
            # Note: shuffle in sklearn.utils can handle multiple arrays and shuffle them consistently
            xtrain_np, ytrain_np, ytrain_label_shuffled = shuffle(xtrain_np, ytrain_np, ytrain_label)

            for xdata, ydata, ind in zip(reversed(xtrain_np), reversed(ytrain_np), range(len(xtrain_np))):
                # xdata and ydata are already numpy arrays from xtrain_np, ytrain_np
                self.forward_propogate(xdata)
                self.back_propogate(ydata)
                if ind % 5 == 0: # Preserving the original logic
                    self.update_weights(eta, gamma / len(xtrain_np)) # Pass gamma regularization term
            
            if test_while_train:
                print('Test Data:')
                tst = self.test(xtest_np, ytest_label) # Use numpy version of xtest
                print('Train Data')
                trn = self.test(xtrain_np, ytrain_label_shuffled) # Use shuffled labels for consistency
                
                if maxaccur <= tst:
                    maxaccur = tst
                    # Store copies of weights
                    maxlayers_hidden_weights = [layer['weights'].copy() for layer in self.hidden_layers]
                    maxlayers_output_weights = self.output_layer['weights'].copy()
                print(maxaccur)
                
            if draw_plot:
                epoch_n.append(i)
                accurtrain.append(trn)
                accurtest.append(tst)
        if draw_plot:
            plt.title('Accuracy w.r.t test data')
            plt.xlabel('No. of Epoch')
            plt.ylabel('Accuracy')
            plt.plot(epoch_n,accurtest)
            plt.savefig('Accur_train2'+str(eta)+'.png')
            
            plt.title('Accuracy w.r.t train data')
            plt.xlabel('No. of Epoch')
            plt.ylabel('Accuracy')
            plt.plot(epoch_n,accurtrain)
            plt.savefig('Accur_train2'+str(eta)+'.png')
        
        
        # Restore the best weights if they were stored
        if test_while_train and maxlayers_output_weights is not None: # Check if weights were stored
            for i, weights_copy in enumerate(maxlayers_hidden_weights):
                self.hidden_layers[i]['weights'] = weights_copy # Use the copied weights
            self.output_layer['weights'] = maxlayers_output_weights # Use the copied weights
        
        # Ensure xtest is numpy array for predict
        xtest_np_final = numpy.array(xtest) if xtest and not isinstance(xtest, numpy.ndarray) else xtest
        if xtest_np_final is not None:
             ypred_labels = self.predict(xtest_np_final)
             print('\n******************\nCalculating for maximum accuracy:**************************')
             max_accuracy = mt.accuracy_score(ytest_label,ypred_labels )
             max_recall = mt.recall_score(ytest_label, ypred_labels, average = None)
             max_fscore = mt.f1_score(ytest_label, ypred_labels, average = None)
             max_precision = mt.precision_score(ytest_label, ypred_labels, average = None)
             print('Accuracy:',max_accuracy,'\nRecall:',max_recall,'\nPrecision:',max_precision,'\nF Score:',max_fscore)
             #print('Layers descpitions') # May print large arrays
             #print('Output Layer Weights:',self.output_layer['weights'])
             #print('Hidden Layer Weights:', [layer['weights'] for layer in self.hidden_layers])
        pass
    
    def back_propogate(self, ydata):
        ydata = numpy.array(ydata) if not isinstance(ydata, numpy.ndarray) else ydata

        # Output Layer derivative (softmax with cross-entropy)
        self.output_layer['derivatives'] = self.output_layer['outputs'] - ydata
        
        # Hidden Layers (iterating backwards)
        next_layer_derivatives = self.output_layer['derivatives']
        next_layer_weights = self.output_layer['weights']

        for layer in reversed(self.hidden_layers):
            # error_signal = (next_layer_weights without bias column).T @ next_layer_derivatives
            error_signal = next_layer_weights[:, :-1].T @ next_layer_derivatives
            
            # layer_derivatives = error_signal * sigmoid_derivative(layer_outputs)
            layer['derivatives'] = error_signal * layer['outputs'] * (1.0 - layer['outputs'])
            
            # Update for the next iteration (previous layer in backprop)
            next_layer_derivatives = layer['derivatives']
            next_layer_weights = layer['weights']
        pass
    
    def predict(self,xdata):
        # xdata should be a numpy array of samples or a list of numpy arrays
        xdata_np = numpy.array(xdata) if not isinstance(xdata, numpy.ndarray) else xdata
        ypred = []
        for x_instance in xdata_np:
            ypred.append(self.predict_per_data(x_instance))
        return ypred
    
    def predict2(self,xdata): # Assuming this is for returning raw probabilities
        # xdata should be a numpy array of samples or a list of numpy arrays
        xdata_np = numpy.array(xdata) if not isinstance(xdata, numpy.ndarray) else xdata
        ypred_probs = []
        for x_instance in xdata_np:
            ypred_probs.append(self.predict_per_data2(x_instance))
        return ypred_probs
    
    def test(self,xdata, ydata_label):
        # xdata should be numpy array
        xdata_np = numpy.array(xdata) if not isinstance(xdata, numpy.ndarray) else xdata
        ypred_label = self.predict(xdata_np)
        accuracy = 0
        for ypred, ytrue in zip(ypred_label, ydata_label):
            if ypred==ytrue:
                accuracy+=1
        accuracy = accuracy/(1.0*len(ydata_label))
        print('Accuracy:',accuracy)
        return accuracy
    
    def predict_per_data(self,x): # x is a single numpy array instance
        probabilities = self.forward_propogate(x)
        return numpy.argmax(probabilities)
    
    def predict_per_data2(self,x): # x is a single numpy array instance
        return self.forward_propogate(x) # Returns raw probabilities

    def update_weights(self, eta, gamma): # gamma is regularization parameter
        # Output Layer
        # Input to output layer is the output (with bias) of the last hidden layer
        # Or, if no hidden layers, it's the input layer's output (with bias)
        if self.hidden_layers:
            inputs_to_output_layer_with_bias = self.hidden_layers[-1]['outputs_with_bias']
        else:
            inputs_to_output_layer_with_bias = self.input_layer_outputs_with_bias
        
        gradient_penalty_output = 2 * gamma * self.output_layer['weights']
        # Note: The derivative term in outer product should be (num_classes, 1) and input (1, num_inputs_to_layer+1)
        # numpy.outer correctly handles (N,) and (M,) to produce (N, M)
        delta_weights_output = eta * (numpy.outer(self.output_layer['derivatives'], inputs_to_output_layer_with_bias) + gradient_penalty_output)
        self.output_layer['weights'] -= delta_weights_output
        self.output_layer['derivatives'].fill(0) # Reset derivatives

        # Hidden Layers (iterating backwards)
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            current_layer = self.hidden_layers[i]
            
            # Determine the input to this layer for delta calculation
            if i == 0: # First hidden layer
                inputs_for_delta_with_bias = self.input_layer_outputs_with_bias
            else: # Subsequent hidden layers
                inputs_for_delta_with_bias = self.hidden_layers[i-1]['outputs_with_bias']
            
            gradient_penalty_hidden = 2 * gamma * current_layer['weights']
            delta_weights_hidden = eta * (numpy.outer(current_layer['derivatives'], inputs_for_delta_with_bias) + gradient_penalty_hidden)
            current_layer['weights'] -= delta_weights_hidden
            current_layer['derivatives'].fill(0) # Reset derivatives


if __name__=='__main__':
    # Configuration
    input_features = 784
    hidden_configs = [[128, 64], [256]] # Test with different hidden layer configurations
    output_classes = 10
    
    # Dummy data
    xtrain_dummy = numpy.random.rand(100, input_features)
    ytrain_labels_dummy = numpy.random.randint(0, output_classes, 100)
    ytrain_one_hot_dummy = numpy.zeros((100, output_classes))
    for i, label in enumerate(ytrain_labels_dummy):
        ytrain_one_hot_dummy[i, label] = 1
        
    xtest_dummy = numpy.random.rand(20, input_features)
    ytest_labels_dummy = numpy.random.randint(0, output_classes, 20)

    for hidden_layer_neurons in hidden_configs:
        print(f"\nTesting with Network Configuration: Input={input_features}, Hidden={hidden_layer_neurons}, Output={output_classes}")
        nn_instance = neural_network(input_features, hidden_layer_neurons, output_classes)

        # Test forward propagation
        print("Testing forward_propagate with one instance...")
        single_instance = xtrain_dummy[0]
        output_probs = nn_instance.forward_propogate(single_instance)
        print("Output probabilities (first 5):", output_probs[:5])
        print("Predicted class:", numpy.argmax(output_probs))

        # Test backpropagation
        print("\nTesting back_propagate...")
        target_output_one_hot = ytrain_one_hot_dummy[0]
        nn_instance.back_propogate(target_output_one_hot)
        print("Derivatives in output layer (first 5):", nn_instance.output_layer['derivatives'][:5])
        if nn_instance.hidden_layers:
            print("Derivatives in last hidden layer (first 5):", nn_instance.hidden_layers[-1]['derivatives'][:5])

        # Test weight update
        print("\nTesting update_weights...")
        learning_rate = 0.01
        regularization_gamma = 0.001 # Example gamma
        
        old_output_weights_sample = nn_instance.output_layer['weights'][0, :5].copy()
        nn_instance.update_weights(learning_rate, regularization_gamma)
        print("Old output weights (sample):", old_output_weights_sample)
        print("New output weights (sample after update):", nn_instance.output_layer['weights'][0, :5])

        # Test training loop (minimal)
        print("\nTesting train loop (1 epoch, minimal data)...")
        # nn_instance.train(xtrain_dummy, ytrain_one_hot_dummy, ytrain_labels_dummy,
        #                   eta=learning_rate, gamma=regularization_gamma, n_epoch=1,
        #                   test_while_train=True, xtest=xtest_dummy, ytest_label=ytest_labels_dummy,
        #                   draw_plot=False)
        # Temporarily simplify train call for quick check without full dataset pass
        nn_instance.train(xtrain_dummy[:10], ytrain_one_hot_dummy[:10], ytrain_labels_dummy[:10],
                          eta=learning_rate, gamma=regularization_gamma, n_epoch=1,
                          test_while_train=False, draw_plot=False)

        print(f"Minimal training test completed for config: {hidden_layer_neurons}")

    print("\nAll basic mechanics tested for different configurations.")
    
