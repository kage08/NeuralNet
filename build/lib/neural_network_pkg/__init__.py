# src/neural_network_pkg/__init__.py

# Import the neural_network class from neural_net.py
from .neural_net import neural_network as NeuralNetworkV1

# Import the neural_network class from neural_net2.py
# Note: Both files define a class named 'neural_network'.
# We need to alias them to avoid conflict if a user wants to access both.
from .neural_net2 import neural_network as NeuralNetworkV2

# Optionally, define __all__ to specify what '*' imports
__all__ = ['NeuralNetworkV1', 'NeuralNetworkV2']

# You could also choose a primary one to expose without an alias if desired,
# e.g., if NeuralNetworkV2 is the recommended one:
# from .neural_net2 import neural_network
# __all__ = ['neural_network', 'NeuralNetworkV1']
# For now, let's stick to distinct aliases.

__version__ = "0.1.0" # Retaining the version from the previous file content
