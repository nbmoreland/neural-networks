# Neural Network Classification with Keras

A TensorFlow/Keras implementation of a configurable neural network for classification tasks on UCI datasets. This project provides a flexible framework for building and evaluating multi-layer neural networks with customizable architecture.

## Author
Nicholas Moreland

## Features

- **Configurable Architecture**: Specify the number of layers and units per layer
- **Automatic Data Normalization**: Input features are normalized by maximum absolute value
- **Tie-Aware Classification**: Handles prediction ties by distributing accuracy credit
- **Detailed Performance Reporting**: Per-instance classification results with accuracy metrics
- **UCI Dataset Support**: Built to work with UCI Machine Learning Repository datasets

## Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- uci_data module (for loading UCI datasets)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/neural-networks.git
cd neural-networks
```

2. Install required dependencies:
```bash
pip install tensorflow numpy
```

3. Ensure the `uci_data` module is available in your Python path. This module should provide the `read_uci1()` function for loading UCI datasets.

## Usage

### Basic Usage

The main function `nn_keras()` accepts the following parameters:

```python
nn_keras(directory, dataset, layers, units_per_layer, epochs)
```

#### Parameters:
- `directory` (str): Path to the directory containing the UCI dataset
- `dataset` (str): Name of the UCI dataset to load
- `layers` (int): Total number of layers in the network (including input and output layers)
- `units_per_layer` (int): Number of neurons in each hidden layer
- `epochs` (int): Number of training epochs

### Example

```python
from nn_keras import nn_keras

# Train a 4-layer network with 20 units per hidden layer for 100 epochs
nn_keras('./data', 'iris', layers=4, units_per_layer=20, epochs=100)
```

### Running from Command Line

```bash
python nn_keras.py
```

Note: You'll need to modify the script to add command-line argument parsing or directly call the function with your desired parameters.

## Architecture Details

### Network Structure
- **Input Layer**: Automatically sized based on the dataset features
- **Hidden Layers**: Configurable number of layers with sigmoid activation
- **Output Layer**: Number of neurons equals the number of classes, using sigmoid activation

### Training Configuration
- **Optimizer**: Adam optimizer
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

### Data Preprocessing
1. **Normalization**: All input features are divided by the maximum absolute value found in the training set
2. **Label Encoding**: Labels are expected to be integer class indices (0 to n-1 for n classes)

## Implementation Details

### Key Features

1. **Dynamic Architecture Creation**
   - Input shape is automatically determined from the training data
   - Number of output classes is calculated from the maximum label value

2. **Tie Handling in Classification**
   - When multiple classes have the same output probability (ties), accuracy credit is divided equally
   - For example, if 3 classes tie for the highest probability and one is correct, accuracy = 1/3

3. **Detailed Output**
   - Each test instance prints: ID, predicted class, true class, and accuracy
   - Final classification accuracy is reported as the average across all test instances

### Code Structure

```
neural-networks/
├── nn_keras.py          # Main neural network implementation
├── CLAUDE.md           # Development guide for Claude AI
└── README.md           # This file
```

## Performance Evaluation

The model evaluates performance using a custom accuracy calculation:

1. For each test instance:
   - Predicts class probabilities
   - Identifies the class with highest probability
   - Checks for ties (multiple classes with same highest probability)
   - Calculates accuracy based on whether true class is among the predictions

2. Overall accuracy is the mean of individual instance accuracies

## Limitations and Considerations

- **Activation Function**: Uses sigmoid activation for all layers, which may not be optimal for all datasets
- **Output Layer**: Sigmoid activation on output layer is unconventional for multi-class classification (softmax is typically preferred)
- **Batch Training**: Currently uses default batch size; consider tuning for large datasets
- **Validation Set**: No validation split is used during training; consider adding for better generalization monitoring

## Future Improvements

Consider these enhancements for production use:

1. **Add Command-Line Interface**: Implement argparse for easier script execution
2. **Activation Functions**: Make activation functions configurable (ReLU, tanh, etc.)
3. **Output Layer**: Use softmax activation for multi-class classification
4. **Early Stopping**: Implement early stopping based on validation loss
5. **Cross-Validation**: Add k-fold cross-validation support
6. **Model Saving**: Add functionality to save and load trained models
7. **Hyperparameter Tuning**: Implement grid search or random search for optimal parameters
8. **Visualization**: Add training history plots and confusion matrices

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is provided as-is for educational purposes. Please add an appropriate license file if planning to use in production.

## Acknowledgments

- UCI Machine Learning Repository for providing the datasets
- TensorFlow/Keras team for the deep learning framework