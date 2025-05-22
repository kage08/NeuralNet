import random
import numpy
from math import exp
from matplotlib import pyplot as plt
from sklearn import metrics as mt
import copy
from sklearn.utils import shuffle

class neural_network:
    def __init__(self, input_layer_n, n_hidden_layers, num_classes):
        random.seed(1)
        numpy.random.seed(1)  # Seed numpy's random number generator as well
        self.input_layer_n = input_layer_n
        self.n_hidden_layers_counts = n_hidden_layers  # Store the counts of neurons
        self.num_classes = num_classes
        
        self.hidden_layers = []
        prev_num_neurons = input_layer_n
        
        # Initializing weights for hidden layers
        for num_neurons in self.n_hidden_layers_counts:
            layer = {
                'weights': numpy.random.rand(num_neurons, prev_num_neurons + 1),
                'derivatives': numpy.zeros(num_neurons),
                'outputs': numpy.zeros(num_neurons)  # To store activations
            }
            self.hidden_layers.append(layer)
            prev_num_neurons = num_neurons
            
        # Initializing weights for output layer
        self.output_layer = {
            'weights': numpy.random.rand(num_classes, prev_num_neurons + 1),
            'derivatives': numpy.zeros(num_classes),
            'outputs': numpy.zeros(num_classes)  # To store activations
        }
        
    def forward_propogate(self, x):
       # Ensure x is a numpy array
       inputs = numpy.array(x) if not isinstance(x, numpy.ndarray) else x
       self.input_layer_values = inputs # Store input layer values

       current_inputs = inputs
       for layer in self.hidden_layers:
           # Concatenate 1 for bias
           current_inputs_with_bias = numpy.concatenate((current_inputs, [1]))
           # Calculate activations: activations = weight_matrix @ input_vector_with_bias
           activations = layer['weights'] @ current_inputs_with_bias
           # Apply sigmoid function element-wise
           layer_outputs = self.sigmoid(activations)
           # Store these layer_outputs
           layer['outputs'] = layer_outputs
           current_inputs = layer_outputs

       # Output layer
       # Concatenate 1 to the activations from the last hidden layer
       last_hidden_outputs_with_bias = numpy.concatenate((current_inputs, [1]))
       # Calculate weighted sums: output_activations = output_weight_matrix @ last_hidden_layer_outputs_with_bias
       output_activations = self.output_layer['weights'] @ last_hidden_outputs_with_bias
       # Apply softmax to output_activations
       final_outputs = self.softmax(output_activations)
       # Store these final outputs
       self.output_layer['outputs'] = final_outputs
       return final_outputs

    def sigmoid(self, x):
        # Ensure x is a numpy array for element-wise operations
        x = numpy.array(x) if not isinstance(x, numpy.ndarray) else x
        return 1.0 / (1.0 + numpy.exp(-x))

    def softmax(self, inp):
        # Ensure inp is a numpy array
        inp = numpy.array(inp) if not isinstance(inp, numpy.ndarray) else inp
        exp_inp = numpy.exp(inp - numpy.max(inp))  # Subtract max for numerical stability
        return exp_inp / numpy.sum(exp_inp)

    def train(self, xtrain, ytrain,ytrain_label, eta,n_epoch,test_while_train=False,xtest=list(), ytest_label=list(),draw_plot = False):
        epoch_n=list()
        accurtrain=list()
        accurtest = list()
        
        maxlayers = list()
        maxaccur = self.test(xtrain, ytrain_label)
        #Training for epoch via stochastic gradient descent
        for i in range(n_epoch):
            print('Epoch No:',i)
            #Shuffle train data
            xtrain, ytrain, ytrain_label = shuffle(xtrain, ytrain, ytrain_label)

            #Training for dataset
            for xdata, ydata in zip(reversed(xtrain), reversed(ytrain)):
                # Ensure xdata and ydata are numpy arrays
                xdata_np = numpy.array(xdata) if not isinstance(xdata, numpy.ndarray) else xdata
                ydata_np = numpy.array(ydata) if not isinstance(ydata, numpy.ndarray) else ydata
                self.forward_propogate(xdata_np)
                self.back_propogate(ydata_np)
                self.update_weights(eta)
            
            #Note down accuracy
            if test_while_train:
                print('Test Data:')
                tst = self.test(xtest, ytest_label)
                print('Train Data')
                trn = self.test(xtrain, ytrain_label)
                
                #Calculating for maximum accuracy
                if maxaccur<=tst:
                    maxaccur=tst
                    # Deepcopy needs to handle numpy arrays correctly.
                    # Storing weights directly, not the entire layer dicts if they contain non-serializable items
                    maxlayers_hidden_weights = [layer['weights'].copy() for layer in self.hidden_layers]
                    maxlayers_output_weights = self.output_layer['weights'].copy()
                    # If you also need to store derivatives or outputs for some reason:
                    # maxlayers_hidden_outputs = [layer['outputs'].copy() for layer in self.hidden_layers]
                    # maxlayers_output_outputs = self.output_layer['outputs'].copy()

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
            plt.savefig('Accur_train'+str(eta)+'.png')
            
            plt.title('Accuracy w.r.t train data')
            plt.xlabel('No. of Epoch')
            plt.ylabel('Accuracy')
            plt.plot(epoch_n,accurtrain)
            plt.savefig('Accur_train'+str(eta)+'.png')
        
        
        # Restore the best weights
        if test_while_train and maxlayers_hidden_weights and maxlayers_output_weights:
            for i, weights in enumerate(maxlayers_hidden_weights):
                self.hidden_layers[i]['weights'] = weights
            self.output_layer['weights'] = maxlayers_output_weights

        ypred_labels = self.predict(xtest)
        print('\n******************\nCalculating for maximum accuracy:**************************')

        max_accuracy = mt.accuracy_score(ytest_label,ypred_labels )
        max_recall = mt.recall_score(ytest_label, ypred_labels, average = None)
        max_fscore = mt.f1_score(ytest_label, ypred_labels, average = None)
        max_precision = mt.precision_score(ytest_label, ypred_labels, average = None)
        print('Accuracy:',max_accuracy,'\nRecall:',max_recall,'\nPrecision:',max_precision,'\nF Score:',max_fscore)
        #print('Layers descpitions') # This might print large arrays, consider summarizing
        #print('Output Layer Weights:',self.output_layer['weights'])
        #print('Hidden Layer Weights:', [layer['weights'] for layer in self.hidden_layers])
        
        pass
    
    def back_propogate(self, ydata):
        # Ensure ydata is a numpy array
        ydata = numpy.array(ydata) if not isinstance(ydata, numpy.ndarray) else ydata

        # Output Layer
        # The error derivative for the output layer (using cross-entropy loss with softmax)
        # is output_layer_outputs - ydata.
        self.output_layer['derivatives'] = self.output_layer['outputs'] - ydata
        
        # Hidden Layers (iterating backwards)
        next_layer_derivatives = self.output_layer['derivatives']
        next_layer_weights = self.output_layer['weights']

        for layer in reversed(self.hidden_layers):
            # error_signal = next_layer_weights.T @ next_layer_derivatives (excluding bias column from next_layer_weights)
            # We need to consider only the weights that connect to the neurons, not the bias weights.
            # So, we take all rows and all columns except the last one (bias weight column).
            error_signal = next_layer_weights[:, :-1].T @ next_layer_derivatives
            
            # layer_derivatives = error_signal * layer_outputs * (1.0 - layer_outputs) (derivative of sigmoid)
            layer['derivatives'] = error_signal * layer['outputs'] * (1.0 - layer['outputs'])
            
            # Update for the next iteration
            next_layer_derivatives = layer['derivatives']
            next_layer_weights = layer['weights']
        pass

    #Predict for a dataset
    def predict(self,xdata):
        ypred = list()
        for x_instance in xdata:
            # Ensure x_instance is a numpy array for forward_propagate
            x_instance_np = numpy.array(x_instance) if not isinstance(x_instance, numpy.ndarray) else x_instance
            ypred.append(self.predict_per_data(x_instance_np))
        return ypred
    
    #Test data
    def test(self,xdata, ydata_label):
        # Ensure xdata is a list of lists/numpy arrays for predict method
        xdata_np = [numpy.array(x) if not isinstance(x, numpy.ndarray) else x for x in xdata]
        ypred_label = self.predict(xdata_np)
        accuracy = 0
        for ypred, ytrue in zip(ypred_label, ydata_label):
            if ypred==ytrue:
                accuracy+=1
        accuracy = accuracy/(1.0*len(ydata_label))
        print('Accuracy:',accuracy)
        return accuracy
    
    #Predict for one data instance
    def predict_per_data(self, x):
        # Ensure x is a numpy array for forward_propagate
        x_np = numpy.array(x) if not isinstance(x, numpy.ndarray) else x
        # forward_propogate returns a numpy array of probabilities
        probabilities = self.forward_propogate(x_np)
        # Return the index of the max probability
        return numpy.argmax(probabilities)

    #Update weights
    def update_weights(self, eta):
        # Output Layer
        # last_hidden_layer_outputs_with_bias
        # The input to the output layer was the output of the last hidden layer, with bias.
        if self.hidden_layers: # Check if there are hidden layers
            last_hidden_layer_outputs = self.hidden_layers[-1]['outputs']
        else: # No hidden layers, input layer is directly connected to output layer
            last_hidden_layer_outputs = self.input_layer_values
        last_hidden_layer_outputs_with_bias = numpy.concatenate((last_hidden_layer_outputs, [1]))
        
        delta_weights_output = eta * numpy.outer(self.output_layer['derivatives'], last_hidden_layer_outputs_with_bias)
        self.output_layer['weights'] -= delta_weights_output
        self.output_layer['derivatives'].fill(0) # Reset derivatives

        # Hidden Layers (iterating backwards)
        # The input to the current hidden layer (for weight update) is the output of the previous layer (or input layer)
        # We iterate from the last hidden layer down to the first.
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            current_layer = self.hidden_layers[i]
            # Determine the input to this layer for delta calculation
            if i == 0: # First hidden layer
                inputs_for_delta_raw = self.input_layer_values
            else: # Subsequent hidden layers
                inputs_for_delta_raw = self.hidden_layers[i-1]['outputs']
            
            inputs_for_delta_with_bias = numpy.concatenate((inputs_for_delta_raw, [1]))
            
            delta_weights_hidden = eta * numpy.outer(current_layer['derivatives'], inputs_for_delta_with_bias)
            current_layer['weights'] -= delta_weights_hidden
            current_layer['derivatives'].fill(0) # Reset derivatives


if __name__=='__main__':
    # Example usage:
    # input_layer_n should be the number of features in the input data
    # n_hidden_layers should be a list of integers, e.g., [number_of_neurons_in_layer1, number_of_neurons_in_layer2]
    # num_classes is the number of output classes

    input_features = 10
    hidden_layer_neurons = [5, 4] # Example: 2 hidden layers with 5 and 4 neurons respectively
    output_classes = 3

    # Create dummy data for testing
    # xtrain: list of lists (or numpy arrays)
    # ytrain: list of lists (one-hot encoded) or just list of labels for ytrain_label
    # For ytrain (target vectors for backprop), it should be one-hot encoded if using softmax cross-entropy
    # For ytrain_label (for accuracy calculation), it should be class indices

    xtrain_list = [[random.random() for _ in range(input_features)] for _ in range(100)]
    # Example ytrain (one-hot encoded)
    ytrain_one_hot_list = []
    ytrain_labels_list = []
    for _ in range(100):
        label = random.randint(0, output_classes - 1)
        ytrain_labels_list.append(label)
        one_hot = [0] * output_classes
        one_hot[label] = 1
        ytrain_one_hot_list.append(one_hot)

    # Convert to numpy arrays before passing to train (or handle inside train)
    xtrain_np = numpy.array(xtrain_list)
    ytrain_one_hot_np = numpy.array(ytrain_one_hot_list)
    ytrain_labels_np = numpy.array(ytrain_labels_list) # For accuracy, labels are fine

    # Dummy test data (optional, if test_while_train is True)
    xtest_list = [[random.random() for _ in range(input_features)] for _ in range(20)]
    ytest_labels_list = [random.randint(0, output_classes - 1) for _ in range(20)]
    xtest_np = numpy.array(xtest_list)
    ytest_labels_np = numpy.array(ytest_labels_list)


    NN = neural_network(input_features, hidden_layer_neurons, output_classes)
    
    # Test forward_propagate with a single instance
    print("Testing forward_propagate with one instance:")
    single_instance = xtrain_np[0]
    output_probs = NN.forward_propogate(single_instance)
    print("Output probabilities:", output_probs)
    print("Predicted class:", numpy.argmax(output_probs))

    # To test training, you would call NN.train(...)
    # Example:
    # NN.train(xtrain_np, ytrain_one_hot_np, ytrain_labels_np, 
    #          eta=0.01, n_epoch=10, 
    #          test_while_train=True, xtest=xtest_np, ytest_label=ytest_labels_np,
    #          draw_plot=False) # Set draw_plot to True if matplotlib is correctly configured

    # For a very simple test of the train loop structure:
    print("\nStarting a minimal training loop for 1 epoch (dummy data):")
    NN.train(xtrain_list, ytrain_one_hot_list, ytrain_labels_list, # Pass lists, train handles conversion
             eta=0.01, n_epoch=1,
             test_while_train=False) # No plotting or intermediate testing for this simple run
    print("Minimal training loop finished.")


    # The old main section for quick check:
    # input_layer = [random.random() for x in range(10)]
    # n_hidden_layers = [5, 4]
    # num_classes = 5
    # NN = neural_network(len(input_layer), n_hidden_layers, num_classes)
    # print(NN.forward_propogate(numpy.array(input_layer))) # Pass numpy array
    # xtrain = [[random.random() for x in range(10)] for x in range(100)]
    # ytrain = [[0]*num_classes for _ in range(100)] # Dummy one-hot ytrain
    # for i in range(100):
    #    ytrain[i][random.randint(0,num_classes-1)] = 1
    # ytrain_labels = [numpy.argmax(y) for y in ytrain] # Dummy labels
    # NN.train(xtrain, ytrain, ytrain_labels, eta=0.01, n_epoch=1)


if __name__=='__main__':
    # Configuration
    input_features = 784  # Example: MNIST image (28x28 pixels)
    hidden_layer_neurons = [128, 64]  # Two hidden layers
    output_classes = 10       # Example: Digits 0-9

    # Create an instance of the neural network
    nn_instance = neural_network(input_features, hidden_layer_neurons, output_classes)

    # Generate some dummy data for a single propagation test
    # (In a real scenario, you'd load your dataset)
    dummy_input_sample = numpy.random.rand(input_features)
    dummy_target_output_one_hot = numpy.zeros(output_classes)
    dummy_target_output_one_hot[numpy.random.randint(0, output_classes)] = 1 # Random one-hot target

    print(f"Input features: {input_features}, Hidden layers: {hidden_layer_neurons}, Output classes: {output_classes}")
    print("Shape of a dummy input sample:", dummy_input_sample.shape)
    
    # Test forward propagation
    print("\nTesting forward_propagate...")
    predicted_probabilities = nn_instance.forward_propogate(dummy_input_sample)
    print("Predicted probabilities:", predicted_probabilities)
    print("Sum of probabilities:", numpy.sum(predicted_probabilities))
    print("Predicted class index:", numpy.argmax(predicted_probabilities))

    # Test backpropagation (requires forward_propagate to have been called first)
    print("\nTesting back_propogate...")
    nn_instance.back_propogate(dummy_target_output_one_hot)
    print("Derivatives in output layer (first 5):", nn_instance.output_layer['derivatives'][:5])
    if nn_instance.hidden_layers:
        print("Derivatives in last hidden layer (first 5):", nn_instance.hidden_layers[-1]['derivatives'][:5])

    # Test weight update (requires back_propogate to have been called)
    print("\nTesting update_weights...")
    learning_rate = 0.01
    # Store old weights for comparison
    old_output_weights_sample = nn_instance.output_layer['weights'][0, :5].copy()
    old_hidden_weights_sample = None
    if nn_instance.hidden_layers:
        old_hidden_weights_sample = nn_instance.hidden_layers[0]['weights'][0, :5].copy()

    nn_instance.update_weights(learning_rate)

    print("Old output weights (sample):", old_output_weights_sample)
    print("New output weights (sample after update):", nn_instance.output_layer['weights'][0, :5])
    if nn_instance.hidden_layers and old_hidden_weights_sample is not None:
        print("Old first hidden layer weights (sample):", old_hidden_weights_sample)
        print("New first hidden layer weights (sample after update):", nn_instance.hidden_layers[0]['weights'][0, :5])
    print("Output layer derivatives after update (should be zero):", nn_instance.output_layer['derivatives'][:5])


    # A more complete training example would require a dataset
    # For now, this tests the mechanics.
    # To run the full train method, you'd need to prepare xtrain, ytrain, ytrain_label etc.
    # e.g., from sklearn.datasets import load_digits
    # digits = load_digits()
    # xtrain = digits.data / 16.0 # Normalize
    # ytrain_labels = digits.target
    # from sklearn.preprocessing import OneHotEncoder
    # encoder = OneHotEncoder(sparse_output=False)
    # ytrain_one_hot = encoder.fit_transform(ytrain_labels.reshape(-1, 1))
    # NN = neural_network(xtrain.shape[1], [64], ytrain_one_hot.shape[1])
    # NN.train(xtrain, ytrain_one_hot, ytrain_labels, eta=0.01, n_epoch=10, test_while_train=True, xtest=xtrain, ytest_label=ytrain_labels)
    print("\nBasic mechanics tested. For full training, uncomment and adapt the training data loading in __main__.")
# Main execution block from the original file (commented out to avoid execution during testing here if not desired)
# if __name__=='__main__':
#     input_layer = [random.random() for x in range(10)]
#     n_hidden_layers = [5, 4]
    # num_classes = 5
    
    # NN = neural_network(len(input_layer), n_hidden_layers, num_classes)
    # print(NN.forward_propogate(numpy.array(input_layer))) # Ensure input is numpy array
    # xtrain = [[random.random() for x in range(10)] for x in range(100)]
    # # ytrain should be one-hot encoded if used with softmax cross-entropy
    # ytrain_labels = [random.randint(0,num_classes-1) for _ in range(100)]
    # ytrain_one_hot = numpy.zeros((100, num_classes))
    # for i, label in enumerate(ytrain_labels):
    #    ytrain_one_hot[i, label] = 1
    
    # NN.train(xtrain, ytrain_one_hot, ytrain_labels, eta=0.01, n_epoch=1) # Example call
    
