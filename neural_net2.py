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
    hidden_layers=list()
    output_layer=list()
    
    n_hidden_layers=list()
    num_classes = 0
    
    
    def __init__(self, input_layer_n, n_hidden_layers, num_classes):
        random.seed(1)
        self.input_layer_n=input_layer_n
        self.n_hidden_layers=n_hidden_layers
        self.num_classes = num_classes
        prev_num_neurons = input_layer_n
        
        for num_neurons in self.n_hidden_layers:
            self.hidden_layers.append([dict({'weights':[random.random()/100.0 for x in range(prev_num_neurons+1)],'derivative':0}) for y in range(num_neurons)])
            prev_num_neurons = num_neurons
            
        self.output_layer = [dict({'weights':[random.random()/100.0 for x in range(self.n_hidden_layers[-1]+1)],'derivative':0}) for y in range(num_classes)]
        
    def forward_propogate(self, x):
       inputs = x
       self.input_layer = x
       for layer in self.hidden_layers:
           new_input = list()
           inputs.append(1)
           for neuron in range(len(layer)):
               layer[neuron]['value'] = self.sigmoid(numpy.dot(inputs,layer[neuron]['weights']))
               new_input.append(layer[neuron]['value'])
           inputs = new_input

       new_input = list()
       inputs.append(1)
       for neuron in range(len(self.output_layer)):
           new_input.append(numpy.dot(inputs, self.output_layer[neuron]['weights']))
           #self.output_layer[neuron]['aggregate'] = numpy.dot(inputs, self.output_layer[neuron]['weights'])

       output =  self.softmax(new_input)
       for neuron in range(len(self.output_layer)):
           self.output_layer[neuron]['value'] = output[neuron]
       return output    

       pass
   
    def sigmoid(self,x):
        return 1.0/(1.0+exp(-x))
    
    def softmax(self,inp):
        return numpy.exp(inp)/numpy.sum(numpy.exp(inp))

    def train(self, xtrain, ytrain,ytrain_label, eta,gamma, n_epoch,test_while_train=False,xtest=list(), ytest_label=list(),draw_plot = False):
        epoch_n=list()
        accurtrain=list()
        accurtest = list()
        
        maxlayers = list()
        maxaccur = self.test(xtrain, ytrain_label)
        for i in range(n_epoch):
            print('Epoch No:',i)
            xtrain, ytrain, ytrain_label = shuffle(xtrain, ytrain, ytrain_label)
            for xdata, ydata ,ind in zip(reversed(xtrain), reversed(ytrain), range(len(xtrain))):
                self.forward_propogate(list(xdata))
                self.back_propogate(list(ydata))
                if ind%5==0:
                    self.update_weights(eta,gamma/len(xtrain))
            #self.update_weights(eta,gamma)
            
            if test_while_train:
                print('Test Data:')
                tst = self.test(xtest, ytest_label)
                print('Train Data')
                trn = self.test(xtrain, ytrain_label)
                
                if maxaccur<=tst:
                    maxaccur=tst
                    maxlayers=copy.deepcopy(self.hidden_layers[:])
                    maxlayers.append(copy.deepcopy(self.output_layer[:]))
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
        
        
        
        self.hidden_layers = maxlayers[:-1]
        self.output_layer = maxlayers[-1]
        ypred_labels = self.predict(xtest)
        print('\n******************\nCalculating for maximum accuracy:**************************')

        max_accuracy = mt.accuracy_score(ytest_label,ypred_labels )
        max_recall = mt.recall_score(ytest_label, ypred_labels, average = None)
        max_fscore = mt.f1_score(ytest_label, ypred_labels, average = None)
        max_precision = mt.precision_score(ytest_label, ypred_labels, average = None)
        print('Accuracy:',max_accuracy,'\nRecall:',max_recall,'\nPrecision:',max_precision,'\nF Score:',max_fscore)
        print('Layers descpitions')
        print('Output Layer:',self.output_layer)
        print('Hidden Layer:', self.hidden_layers)
        
        pass
    
    def back_propogate(self, ydata):
        
        for neuron in self.output_layer:
            for neuron2, y in zip(self.output_layer,ydata):
                if neuron2 is neuron:
                    neuron['derivative']+= (y - neuron2['value'])*neuron['value']*(neuron['value']-1)
                else:
                    neuron['derivative']+= (y - neuron2['value'])*neuron['value']*neuron2['value']
        
        prev_layer = self.output_layer    
        for layer in reversed(self.hidden_layers):
            for neuron, neuron_index in zip(layer, range(len(layer))):
                out_error = 0
                for out_neuron in prev_layer:
                    out_error+= out_neuron['derivative']*out_neuron['weights'][neuron_index]
                neuron['derivative']+=(out_error*neuron['value']*(1.0-neuron['value']))
            prev_layer = layer
        pass
    
    def predict(self,xdata):
        ypred = list()
        for x in xdata:
            ypred.append(self.predict_per_data(list(x)))
        return ypred
    
    def predict2(self,xdata):
        ypred = list()
        for x in xdata:
            ypred.append(self.predict_per_data2(list(x)))
        return ypred
    
    def test(self,xdata, ydata_label):
        ypred_label = self.predict(xdata)
        accuracy = 0
        for ypred, ytrue in zip(ypred_label, ydata_label):
            if ypred==ytrue:
                accuracy+=1
        accuracy = accuracy/(1.0*len(ydata_label))
        print('Accuracy:',accuracy)
        return accuracy
    
    def predict_per_data(self,x):
        y = list(self.forward_propogate(list(x)))
        return y.index(max(y))
    
    def predict_per_data2(self,x):
        y = list(self.forward_propogate(list(x)))
        return y

    def update_weights(self, eta,gamma):
        layer_below_index = len(self.hidden_layers)-1
        for neuron in self.output_layer:
            for windex in range(len(neuron['weights'])-1):
                neuron['weights'][windex] -= eta*(neuron['derivative'])*self.hidden_layers[layer_below_index][windex]['value'] + eta*2*gamma*neuron['weights'][windex]
            neuron['weights'][-1] -= eta*(neuron['derivative'])
            neuron['derivative']= 0
            
            
        layer_below_index -=1
        for layer in self.hidden_layers:
            for neuron in layer:
                if layer_below_index>=0:
                    for windex in range(len(neuron['weights'])-1):
                        neuron['weights'][windex] -= eta*(neuron['derivative'])*self.hidden_layers[layer_below_index][windex]['value']  + eta*2*gamma*neuron['weights'][windex]
                    neuron['weights'][-1] -= eta*(neuron['derivative'])
                else:
                    for windex in range(len(neuron['weights'])-1):
                        neuron['weights'][windex] -= eta*(neuron['derivative'])*self.input_layer[windex]+ eta*2*gamma*neuron['weights'][windex]
                    neuron['weights'][-1] -= eta*(neuron['derivative'])
                neuron['derivative'] = 0
                



if __name__=='__main__':
    input_layer = [random.random() for x in range(10)]
    n_hidden_layers = [5, 4]
    num_classes = 5
    
    NN = neural_network(len(input_layer), n_hidden_layers, num_classes)
    print(NN.forward_propogate(input_layer))
    xtrain = [[random.random() for x in range(10)] for x in range(100)]
    ytrain = [[random.random() for x in range(5)] for x in range(100)]
    #NN.train(xtrain, ytrain,0.01)
    
