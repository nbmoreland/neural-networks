# Nicholas Moreland

from uci_data import *
import tensorflow as tf
import numpy as np

def nn_keras(directory, dataset, layers, units_per_layer, epochs):
    # Load the dataset
    (training_set, test_set) = read_uci1(directory, dataset)
    (training_inputs, training_labels) = training_set
    (test_inputs, test_labels) = test_set
    
    # Normalize input data
    max_value = np.max(np.abs(training_inputs))
    training_inputs = training_inputs / max_value
    test_inputs = test_inputs / max_value

    # Creating the model
    input_shape = training_inputs[0].shape
    number_of_classes = np.max([np.max(training_labels), np.max(test_labels)]) + 1

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    # Add hidden layers with the sigmoid activation function
    for _ in range(layers - 2):
        model.add(tf.keras.layers.Dense(units_per_layer, activation='sigmoid'))

    model.add(tf.keras.layers.Dense(number_of_classes, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Training the model
    model.fit(training_inputs, training_labels, epochs=epochs)

    # Classification on test data
    classification_accuracy = 0
    for object_id in range(len(test_inputs)):
        input_vector = np.reshape(test_inputs[object_id], (1, -1))
        nn_output = model.predict(input_vector)
        nn_output = nn_output.flatten()

        predicted_class = np.argmax(nn_output)
        true_class = test_labels[object_id]

        # Check for ties in classification result
        indices = np.where(nn_output == nn_output[predicted_class])[0]
        number_of_ties = len(indices)

        if predicted_class == true_class:
            if number_of_ties == 1:
                accuracy = 1.0
            else:
                accuracy = 1.0 / number_of_ties
        else:
            accuracy = 0.0

        # Print classification information
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % (object_id + 1, predicted_class, true_class, accuracy))
        classification_accuracy += accuracy

    # Print accuracy
    classification_accuracy /= len(test_inputs)
    print('classification accuracy=%6.4f' % classification_accuracy)
