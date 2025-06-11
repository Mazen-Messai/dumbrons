//
// Created by Mazen Messai on 09/06/2025.
//
// This file is released under the MIT License.
//

#include "layer.h"
#include <random>

// Bias and weights are initialized, inputs and outputs are initialized to zero
Layer::Layer(size_t in_size, size_t out_size,
             std::function<double(double)> activation,
             std::function<double(double)> activation_deriv)
    : weights(out_size, in_size),
      biases(out_size, 1, 0.0),
      inputs(in_size, 1, 0.0),
      outputs(out_size, 1, 0.0),
      deltas(out_size, 1, 0.0),
      activation(activation),
      activation_deriv(activation_deriv)
{
    // Initialize weights with random values in the range [-1, 1]
    // Using a random device and Mersenne Twister for better randomness
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    // Fill the weights matrix with random values
    for (size_t i = 0; i < out_size; ++i) {
        for (size_t j = 0; j < in_size; ++j) {
            weights(i, j) = dist(gen);
        }
    }
}

Matrix Layer::forward(const Matrix& input) {
    // Just some checks to ensure the input is a column vector of the correct size
    if (input.numRows() != weights.numCols() || input.numCols() != 1) {
        throw std::invalid_argument("Input must be a column vector of size (in, 1)");
    }

    // Reset outputs and deltas
    inputs = input;

    // Compute the linear combination of inputs and weights, plus biases
    // In other words, it computes z = W * x + b
    Matrix z = weights * input + biases;

    // Store the outputs of the layer after applying the activation function
    outputs = z;

    // Apply the activation function to each element of the outputs
    // I think the structure  should be improved to make using activation functions like softmax easier
    // For now, we assume the activation function is applied element-wise
    for (size_t i = 0; i < outputs.numRows(); ++i) {
        outputs(i, 0) = activation(outputs(i, 0));
    }

    return outputs;
}

Matrix Layer::backward(const Matrix& dLoss_dOutput) {
    if (dLoss_dOutput.numRows() != outputs.numRows() || dLoss_dOutput.numCols() != 1) {
        throw std::invalid_argument("dLoss/dOutput must match output dimensions");
    }

    // Reset deltas
    deltas = outputs;

    // Compute the deltas for the layer
    // deltas = dActivation(outputs) * dLoss/dOutput
    for (size_t i = 0; i < deltas.numRows(); ++i) {
        deltas(i, 0) = activation_deriv(outputs(i, 0)) * dLoss_dOutput(i, 0);
    }

    // Compute the gradient of the loss with respect to the weights
    // dLoss/dWeights = deltas * inputs^T
    Matrix dLoss_dInput = weights.transpose() * deltas;

    return dLoss_dInput;
}

void Layer::update(double learning_rate) {
    // Update the weights and biases using the deltas computed during backpropagation
    // The gradient of the loss with respect to the weights is given by deltas * inputs^T
    // The gradient of the loss with respect to the biases is given by deltas

    Matrix grad_w = deltas * inputs.transpose();

    for (size_t i = 0; i < weights.numRows(); ++i) {
        for (size_t j = 0; j < weights.numCols(); ++j) {
            // Update the weights using gradient descent
            // w_i,j = w_i,j - learning_rate * dLoss/dWeights_i,j
            weights(i, j) -= learning_rate * grad_w(i, j);
        }
    }

    for (size_t i = 0; i < biases.numRows(); ++i) {
        // Update the biases using gradient descent
        // b_i = b_i - learning_rate * dLoss/dBias_i
        biases(i, 0) -= learning_rate * deltas(i, 0);
    }
}