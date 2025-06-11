//
// Created by Mazen Messai on 09/06/2025.
//
// This file is released under the MIT License.
//

#include "network.h"

Network::Network(const std::vector<size_t>& sizes,
                const std::vector<std::function<double(double)>>& activations,
                const std::vector<std::function<double(double)>>& activation_deriv,
                 std::function<double(double, double)> cost,
                 std::function<double(double, double)> cost_deriv,
                 double learning_rate)
    : learning_rate(learning_rate), cost(cost), cost_deriv(cost_deriv) {
    // Check if the sizes vector is valid
    // It should contain at least 3 elements (number of hidden layers + input and output layers)
    if (sizes.size() < 2) {
        throw std::invalid_argument("Network must have at least an input and an output layer");
    }

    // Create the layers based on the sizes and activation functions provided
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        layers.emplace_back(sizes[i], sizes[i+1], activations[i], activation_deriv[i]);
    }
}

Matrix Network::forward(const Matrix &input) {
    Matrix out = input;

    // Forward has already been implemented in the Layer class
    // Here we just call the forward method of each layer in sequence
    for (size_t i = 0; i < layers.size(); ++i) {
        out = layers[i].forward(out);
    }

    return out;
}

void Network::train(const std::vector<Matrix>& inputs,
                    const std::vector<Matrix>& targets,
                    size_t epochs) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same size");
    }

    // The following code implements the training loop
    // For each epoch, we iterate over all inputs and targets
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle the inputs and targets together
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass through the network
            Matrix out = forward(inputs[i]);

            // Compute the loss gradient using the cost derivative
            // This assumes the cost function is differentiable and returns a gradient
            // For simplicity, we assume the cost function is mean squared error (MSE)
            Matrix loss_grad(out.numRows(), 1);
            for (size_t j = 0; j < out.numRows(); ++j) {
                double y_pred = out(j, 0);
                double y_true = targets[i](j, 0);
                loss_grad(j, 0) = cost_deriv(y_pred, y_true);
            }

            // Backpropagation through the network
            // We start from the output layer and propagate the gradients back through each layer
            // The backward method of each layer computes the gradient of the loss with respect to the inputs
            // and returns it to be used in the previous layer
            Matrix grad = loss_grad;
            for (int l = layers.size() - 1; l >= 0; --l) {
                grad = layers[l].backward(grad);
            }

            // Update the weights and biases of each layer using the computed gradients
            // The update method of each layer applies the gradients to the weights and biases
            for (size_t l = 0; l < layers.size(); ++l) {
                layers[l].update(learning_rate);
            }
        }
    }
}
