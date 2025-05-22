import unittest
import numpy as np

# Assuming neural_net.py and neural_net2.py are in the same directory or accessible in PYTHONPATH
from neural_net import neural_network as nn1
from neural_net2 import neural_network as nn2

class TestNeuralNetworks(unittest.TestCase):

    def _test_initialization(self, network_class, network_name):
        input_n = 3
        hidden_config = [4, 2]
        classes_n = 3
        if network_name == "nn1":
            net = network_class(input_layer_n=input_n, n_hidden_layers=hidden_config, num_classes=classes_n)
        else: # nn2
            net = network_class(input_layer_n=input_n, n_hidden_layers_counts=hidden_config, num_classes=classes_n)

        # Test hidden layers
        self.assertEqual(len(net.hidden_layers), len(hidden_config), f"{network_name}: Incorrect number of hidden layers")
        prev_neurons = input_n
        for i, num_neurons in enumerate(hidden_config):
            layer = net.hidden_layers[i]
            self.assertIsInstance(layer['weights'], np.ndarray, f"{network_name}: Hidden layer {i} weights not ndarray")
            self.assertEqual(layer['weights'].shape, (num_neurons, prev_neurons + 1), f"{network_name}: Hidden layer {i} weights shape mismatch")
            self.assertIsInstance(layer['derivatives'], np.ndarray, f"{network_name}: Hidden layer {i} derivatives not ndarray")
            self.assertEqual(layer['derivatives'].shape, (num_neurons,), f"{network_name}: Hidden layer {i} derivatives shape mismatch")
            np.testing.assert_array_equal(layer['derivatives'], np.zeros(num_neurons), f"{network_name}: Hidden layer {i} derivatives not initialized to zero")
            self.assertIn('outputs', layer, f"{network_name}: Hidden layer {i} missing 'outputs'")
            self.assertIsInstance(layer['outputs'], np.ndarray, f"{network_name}: Hidden layer {i} outputs not ndarray")
            self.assertEqual(layer['outputs'].shape, (num_neurons,), f"{network_name}: Hidden layer {i} outputs shape mismatch")
            if network_name == "nn2": # nn2 specific structure
                 self.assertIn('outputs_with_bias', layer, f"{network_name}: Hidden layer {i} missing 'outputs_with_bias'")
                 self.assertIsInstance(layer['outputs_with_bias'], np.ndarray, f"{network_name}: Hidden layer {i} outputs_with_bias not ndarray")
                 self.assertEqual(layer['outputs_with_bias'].shape, (num_neurons+1,), f"{network_name}: Hidden layer {i} outputs_with_bias shape mismatch")

            prev_neurons = num_neurons

        # Test output layer
        self.assertIsInstance(net.output_layer['weights'], np.ndarray, f"{network_name}: Output layer weights not ndarray")
        self.assertEqual(net.output_layer['weights'].shape, (classes_n, prev_neurons + 1), f"{network_name}: Output layer weights shape mismatch")
        self.assertIsInstance(net.output_layer['derivatives'], np.ndarray, f"{network_name}: Output layer derivatives not ndarray")
        self.assertEqual(net.output_layer['derivatives'].shape, (classes_n,), f"{network_name}: Output layer derivatives shape mismatch")
        np.testing.assert_array_equal(net.output_layer['derivatives'], np.zeros(classes_n), f"{network_name}: Output layer derivatives not initialized to zero")
        self.assertIn('outputs', net.output_layer, f"{network_name}: Output layer missing 'outputs'")
        self.assertIsInstance(net.output_layer['outputs'], np.ndarray, f"{network_name}: Output layer outputs not ndarray")
        self.assertEqual(net.output_layer['outputs'].shape, (classes_n,), f"{network_name}: Output layer outputs shape mismatch")

    def test_nn1_initialization(self):
        self._test_initialization(nn1, "nn1")

    def test_nn2_initialization(self):
        self._test_initialization(nn2, "nn2")

    def _test_forward_propagation(self, network_class, network_name):
        # Simplified network: 1 input, 1 hidden neuron, 2 output classes
        input_n = 1
        hidden_config = [1]
        classes_n = 2
        if network_name == "nn1":
            net = network_class(input_layer_n=input_n, n_hidden_layers=hidden_config, num_classes=classes_n)
        else: # nn2
            net = network_class(input_layer_n=input_n, n_hidden_layers_counts=hidden_config, num_classes=classes_n)

        # Manually set weights for predictable output
        # Hidden layer weights: [w_h_input, w_h_bias]
        net.hidden_layers[0]['weights'] = np.array([[0.5, 0.2]]) # 1 neuron, 1 input + 1 bias
        # Output layer weights: [[w_o1_h, w_o1_bias], [w_o2_h, w_o2_bias]]
        net.output_layer['weights'] = np.array([[0.6, 0.3], [0.4, 0.1]]) # 2 outputs, 1 hidden neuron + 1 bias

        input_data = np.array([0.7])
        output_probs = net.forward_propogate(input_data)

        # Manual calculation:
        # Hidden Layer
        # input_layer_values is [0.7]
        # For nn1, input to hidden is [0.7, 1.0] (inputs.append(1))
        # For nn2, input_layer_outputs_with_bias is [0.7, 1.0]
        h_input_aug = np.array([0.7, 1.0]) 
        h_weighted_sum = net.hidden_layers[0]['weights'] @ h_input_aug # 0.5*0.7 + 0.2*1 = 0.35 + 0.2 = 0.55
        h_output = 1.0 / (1.0 + np.exp(-h_weighted_sum[0])) # sigmoid(0.55) approx 0.634135591015
        
        np.testing.assert_array_almost_equal(net.hidden_layers[0]['outputs'], np.array([h_output]), decimal=5, err_msg=f"{network_name}: Hidden layer output mismatch")
        if network_name == "nn2":
            np.testing.assert_array_almost_equal(net.hidden_layers[0]['outputs_with_bias'], np.array([h_output, 1.0]), decimal=5, err_msg=f"{network_name}: Hidden layer output_with_bias mismatch")
            np.testing.assert_array_almost_equal(net.input_layer_outputs_with_bias, h_input_aug, decimal=5, err_msg=f"{network_name}: Input layer output_with_bias mismatch")


        # Output Layer
        # Input to output layer (augmented): [h_output, 1.0]
        o_input_aug = np.array([h_output, 1.0])
        o_weighted_sum = net.output_layer['weights'] @ o_input_aug 
        # [0.6*h_output + 0.3*1, 0.4*h_output + 0.1*1]
        # [0.6*0.634135591015 + 0.3, 0.4*0.634135591015 + 0.1]
        # [0.380481354609 + 0.3, 0.253654236406 + 0.1]
        # [0.680481354609, 0.353654236406]
        expected_raw_output = np.array([0.680481354609, 0.353654236406])
        
        # Softmax
        exp_raw = np.exp(expected_raw_output - np.max(expected_raw_output))
        expected_softmax_output = exp_raw / np.sum(exp_raw)

        np.testing.assert_array_almost_equal(output_probs, expected_softmax_output, decimal=5, err_msg=f"{network_name}: Final output probabilities mismatch")
        np.testing.assert_array_almost_equal(net.output_layer['outputs'], expected_softmax_output, decimal=5, err_msg=f"{network_name}: Output layer stored outputs mismatch")

    def test_nn1_forward_propagation(self):
        self._test_forward_propagation(nn1, "nn1")

    def test_nn2_forward_propagation(self):
        self._test_forward_propagation(nn2, "nn2")

    def _test_back_propagation(self, network_class, network_name):
        input_n = 1
        hidden_config = [1]
        classes_n = 2
        if network_name == "nn1":
            net = network_class(input_layer_n=input_n, n_hidden_layers=hidden_config, num_classes=classes_n)
        else: # nn2
            net = network_class(input_layer_n=input_n, n_hidden_layers_counts=hidden_config, num_classes=classes_n)

        net.hidden_layers[0]['weights'] = np.array([[0.5, 0.2]])
        net.output_layer['weights'] = np.array([[0.6, 0.3], [0.4, 0.1]])
        
        input_data = np.array([0.7])
        # Perform forward pass to populate outputs
        output_probs = net.forward_propogate(input_data) 
        
        target_y = np.array([1.0, 0.0]) # Example target: class 0 is correct

        # Expected derivatives:
        # Output layer: derivatives = outputs - y_true
        expected_output_derivatives = output_probs - target_y
        
        net.back_propogate(target_y) # Call backprop

        np.testing.assert_array_almost_equal(net.output_layer['derivatives'], expected_output_derivatives, decimal=5, err_msg=f"{network_name}: Output layer derivatives mismatch")

        # Hidden layer derivatives:
        # error_signal = next_layer_weights_no_bias.T @ next_layer_derivatives
        # layer_derivatives = error_signal * layer_outputs * (1 - layer_outputs)
        
        # next_layer_weights_no_bias for output layer connection to hidden layer 0:
        # self.output_layer['weights'][:, :-1] -> [[0.6], [0.4]]
        output_weights_no_bias = net.output_layer['weights'][:, :-1] # Shape (2,1)
        
        # next_layer_derivatives are expected_output_derivatives
        error_signal_hidden = output_weights_no_bias.T @ expected_output_derivatives # (1,2) @ (2,) -> (1,)
        
        h_output = net.hidden_layers[0]['outputs'][0] # Scalar value from the single hidden neuron
        sigmoid_derivative = h_output * (1.0 - h_output)
        expected_hidden_derivative = error_signal_hidden * sigmoid_derivative
        
        np.testing.assert_array_almost_equal(net.hidden_layers[0]['derivatives'], expected_hidden_derivative, decimal=5, err_msg=f"{network_name}: Hidden layer 0 derivatives mismatch")

    def test_nn1_back_propagation(self):
        self._test_back_propagation(nn1, "nn1")
        
    def test_nn2_back_propagation(self):
        # The backprop logic for derivatives is the same for nn1 and nn2 (outputs - ydata for softmax CE)
        self._test_back_propagation(nn2, "nn2")


    def _test_update_weights(self, network_class, network_name, include_regularization=False):
        input_n = 1
        hidden_config = [1]
        classes_n = 2
        if network_name == "nn1":
            net = network_class(input_layer_n=input_n, n_hidden_layers=hidden_config, num_classes=classes_n)
        else: # nn2
            net = network_class(input_layer_n=input_n, n_hidden_layers_counts=hidden_config, num_classes=classes_n)

        # Set known weights
        net.hidden_layers[0]['weights'] = np.array([[0.5, 0.2]], dtype=float)
        net.output_layer['weights'] = np.array([[0.6, 0.3], [0.4, 0.1]], dtype=float)
        
        original_hidden_weights = net.hidden_layers[0]['weights'].copy()
        original_output_weights = net.output_layer['weights'].copy()

        input_data = np.array([0.7])
        target_y = np.array([1.0, 0.0])
        
        # Forward and backward pass to get derivatives and activations
        net.forward_propogate(input_data)
        net.back_propogate(target_y)

        eta = 0.1
        gamma = 0.01 # For nn2
        
        # --- Expected Output Layer Weight Update ---
        # Input to output layer (augmented from hidden layer's output)
        h_output = net.hidden_layers[0]['outputs'][0]
        if network_name == "nn1":
             # In nn1, forward_propogate was modified to store inputs for layers,
             # but update_weights uses self.hidden_layers[layer_below_index][windex]['value']
             # which is h_output. The bias is handled by a separate term.
             # This test needs to align with how update_weights actually gets its inputs.
             # Re-checking nn1.update_weights:
             # neuron['weights'][windex] -= eta*(neuron['derivative'])*self.hidden_layers[layer_below_index][windex]['value']
             # neuron['weights'][-1] -= eta*(neuron['derivative']) (bias term)
             # This structure is different from the vectorized one.
             # The test below assumes vectorized update_weights as refactored.
             # Let's use the refactored logic for expected calculation.
             inputs_to_output_layer_with_bias = np.array([h_output, 1.0])
        elif network_name == "nn2":
             inputs_to_output_layer_with_bias = net.hidden_layers[0]['outputs_with_bias']
        else: # Fallback for nn1 if its structure is indeed different
             inputs_to_output_layer_with_bias = np.array([h_output, 1.0])


        output_derivatives = net.output_layer['derivatives'] # Shape (2,)
        grad_output_weights = np.outer(output_derivatives, inputs_to_output_layer_with_bias) # Shape (2, 2)
        
        expected_new_output_weights = original_output_weights.copy()
        if include_regularization: # For nn2
            penalty_output = 2 * gamma * original_output_weights
            expected_new_output_weights -= eta * (grad_output_weights + penalty_output)
        else: # For nn1
            expected_new_output_weights -= eta * grad_output_weights
            
        # --- Expected Hidden Layer Weight Update ---
        # Input to hidden layer (augmented from network input)
        if network_name == "nn1":
            # Similar to output layer, nn1's original update_weights was different.
            # Using refactored logic for expected calculation.
            inputs_to_hidden_layer_with_bias = np.array([input_data[0], 1.0])
        elif network_name == "nn2":
            inputs_to_hidden_layer_with_bias = net.input_layer_outputs_with_bias
        else: # Fallback for nn1
            inputs_to_hidden_layer_with_bias = np.array([input_data[0], 1.0])


        hidden_derivatives = net.hidden_layers[0]['derivatives'] # Shape (1,)
        grad_hidden_weights = np.outer(hidden_derivatives, inputs_to_hidden_layer_with_bias) # Shape (1, 2)

        expected_new_hidden_weights = original_hidden_weights.copy()
        if include_regularization: # For nn2
            penalty_hidden = 2 * gamma * original_hidden_weights
            expected_new_hidden_weights -= eta * (grad_hidden_weights + penalty_hidden)
        else: # For nn1
            expected_new_hidden_weights -= eta * grad_hidden_weights

        # Perform the update
        if include_regularization:
            net.update_weights(eta, gamma)
        else:
            net.update_weights(eta)

        np.testing.assert_array_almost_equal(net.output_layer['weights'], expected_new_output_weights, decimal=5, err_msg=f"{network_name}: Output layer weights update mismatch")
        np.testing.assert_array_almost_equal(net.hidden_layers[0]['weights'], expected_new_hidden_weights, decimal=5, err_msg=f"{network_name}: Hidden layer weights update mismatch")

        # Verify derivatives are reset
        np.testing.assert_array_equal(net.output_layer['derivatives'], np.zeros(classes_n), f"{network_name}: Output derivatives not reset")
        np.testing.assert_array_equal(net.hidden_layers[0]['derivatives'], np.zeros(hidden_config[0]), f"{network_name}: Hidden derivatives not reset")

    def test_nn1_update_weights(self):
        self._test_update_weights(nn1, "nn1", include_regularization=False)

    def test_nn2_update_weights(self):
        self._test_update_weights(nn2, "nn2", include_regularization=True)


if __name__ == '__main__':
    # unittest.main() # This will run all tests in the module
    # To run from script with specific args if needed (e.g. in some environments)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
