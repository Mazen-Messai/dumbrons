//
// Created by mazen on 09/06/2025.
// This file is part of a simple neural network library for C++.
// It provides a Layer class that represents a single layer in a neural network.
// The Layer class supports forward and backward passes, weight updates, and uses activation functions.
// The library is designed to be easy to use and extend, with a focus on educational purposes.
// I wrote it to practice C++ and have fun with machine learning concepts.
// It is just a simple implementation of a neural network layer, not optimized for performance or memory usage.
//
// This file is released under the MIT License.
//

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "matrix.h"



class Layer {
private:
    // Weights and biases are stored as matrices
    // Weights are of size (out_size, in_size) and biases are of size (out_size, 1)
    // Outputs are of size (out_size, 1) and inputs are of size (in_size, 1)
    // Deltas are of size (out_size, 1) and are used for backpropagation
    // Activation functions are stored as function pointers
    // Activation is a function that takes a double and returns a double
    Matrix weights;
    Matrix biases;
    Matrix outputs;
    Matrix inputs;
    Matrix deltas;

    std::function<double(double)> activation;
    std::function<double(double)> activation_deriv;

public:
    // Constructor to initialize the layer with given sizes and activation functions
    //
    // Parameters :
    // in_size : number of input neurons
    // out_size : number of output neurons
    // activation : activation function to be used in the layer
    // activation_deriv : derivative of the activation function
    Layer(size_t in_size, size_t out_size,
          std::function<double(double)> activation,
          std::function<double(double)> activation_deriv);

    // Forward pass through the layer
    //
    // Parameters :
    // input : the input matrix, should be a column vector of size (in_size, 1)
    // output : the output matrix, which is the result of the forward pass through the layer, should be a column vector of size (out_size, 1)
    //
    // throws std::invalid_argument if the input dimensions do not match the expected size
    Matrix forward(const Matrix& input);

    // Backward pass through the layer
    //
    // Parameters :
    // dLoss_dOutput : the gradient of the loss with respect to the output of the layer, should be a column vector of size (out_size, 1)
    // output : the gradient of the loss with respect to the input of the layer, should be a column vector of size (in_size, 1)
    //
    // throws std::invalid_argument if the dimensions of dLoss_dOutput do not match the output size of the layer
    Matrix backward(const Matrix& dLoss_dOutput);

    // Update the weights and biases of the layer using the deltas computed during backpropagation
    //
    // Parameters :
    // learning_rate : the learning rate to be used for updating the weights and biases
    void update(double learning_rate);

    // Getters for the weights, biases, outputs, inputs, and deltas
    const Matrix& getOutput() const { return outputs; }
    const Matrix& getDelta() const { return deltas; }
};



#endif //LAYER_H
