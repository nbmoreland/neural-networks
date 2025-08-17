# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a neural network implementation project using TensorFlow/Keras for classification tasks on UCI datasets.

## Dependencies
- TensorFlow (with Keras)
- NumPy
- uci_data module (custom module for loading UCI datasets)

## Key Commands

### Running the Neural Network
```bash
python nn_keras.py
```

### Install Dependencies
```bash
pip install tensorflow numpy
```

## Architecture

### Core Components

**nn_keras.py** - Main neural network implementation
- `nn_keras(directory, dataset, layers, units_per_layer, epochs)`: Main function that:
  - Loads UCI dataset using the `uci_data` module
  - Normalizes input data by dividing by the maximum absolute value
  - Creates a sequential Keras model with configurable hidden layers
  - Uses sigmoid activation functions throughout
  - Trains using Adam optimizer with SparseCategoricalCrossentropy loss
  - Evaluates on test set with custom accuracy calculation that handles tied predictions
  - Prints per-instance classification results and overall accuracy

### Data Flow
1. Dataset loaded via `read_uci1()` from the uci_data module
2. Input normalization applied to both training and test sets
3. Model architecture dynamically created based on input shape and number of classes
4. Training performed for specified epochs
5. Test evaluation with detailed per-instance reporting

### Important Implementation Details
- The model uses sigmoid activation for all layers (including output)
- Classification handles ties by dividing accuracy credit equally among tied predictions
- Input normalization uses the maximum absolute value from training data
- Number of output classes determined from the maximum label value in the data