//
// Created by Mazen Messai on 09/06/2025.
// This file is part of a simple neural network library for C++.
// It provides a Network class that represents a neural network composed of multiple layers.
// The Network class supports forward and backward passes, training with a dataset, and uses various activation and cost functions.
// The library is designed to be easy to use and extend, with a focus on educational purposes.
// I wrote it to practice C++ and have fun with machine learning concepts.
// It is just a simple implementation of a neural network, not optimized for performance or memory usage.
//
// This file is released under the MIT License.
//

#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "matrix.h"
#include "layer.h"

class Network {
private:
    std::vector<Layer> layers;
    double learning_rate;
    std::function<double(double, double)> cost;
    std::function<double(double, double)> cost_deriv;

public:
    // Size is a vector of layer sizes, e.g., {2, 3, 1} for a network with 2 input neurons, 3 hidden neurons, and 1 output neuron.
    // For now, the activations and activation_derive are two differents vectors
    // A futur improvement is to generate the activation_deriv automatically with a new Class.
    // Parameters :
    // sizes: vector of layer sizes
    // activations: vector of activation functions for each layer, the activation is a function that takes a double and returns a double
    // activation_deriv: vector of derivative functions for each activation function
    // cost: cost function that takes two doubles (predicted and target) and returns a double
    // cost_deriv: derivative of the cost function that takes two doubles (predicted and target) and returns a double
    // learning_rate: learning rate for the network, default is 0.01
    Network(const std::vector<size_t>& sizes,
            const std::vector<std::function<double(double)>>& activations,
            const std::vector<std::function<double(double)>>& activation_deriv,
            std::function<double(double, double)> cost,
            std::function<double(double, double)> cost_deriv,
            double learning_rate = 0.01);

    // Forward pass through the network
    // Parameters :
    // input : the input matrix, should be a column vector of size (input_size, 1)
    // output : the output matrix, which is the result of the forward pass through the network, should be a column vector of size (output_size, 1)
    Matrix forward(const Matrix& input);

    // Train the network using the provided inputs and targets
    // Parameters :
    // inputs: vector of input matrices, each should be a column vector of size (input_size, 1)
    // targets: vector of target matrices, each should be a column vector of size (output_size, 1)
    // epochs: number of epochs to train the network
    void train(const std::vector<Matrix>& inputs,
               const std::vector<Matrix>& targets,
               size_t epochs);
};


#endif //NETWORK_H
