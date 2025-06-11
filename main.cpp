//
// Created by Mazen Messai on 09/06/2025.
// This file is part of a simple neural network library for C++.
// It runs a neural network on the MNIST dataset for handwritten digit recognition.
// The library is designed to be easy to use and extend, with a focus on educational purposes.
// I wrote it to practice C++ and have fun with machine learning concepts.
// Note that it is my first C++ project, so it is not optimized for performance or memory usage.
// I used it to learn C++ and understand the basics of neural networks.
//
// The sources I used to write this code are:
// - 3blue1brown AMAZING course about Neural networks : https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
// - My machine learning course at university, made by my professors, Jean-Yves Tourneret and Axel Carlier : https://perso.tesa.prd.fr/jyt/ML.html
// - The MNIST dataset
//
// This file is released under the MIT License.
//

#include "network.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <numeric>

int main() {
    // Activation functions and their derivatives
    // Note that the activation functions are defined as lambda functions for simplicity
    // You can replace them with any other activation functions you want to use
    // A future improvement would be to use a class for activation functions and implements softmax for multi-class classification
    auto relu = [](double x) { return x > 0 ? x : 0; };
    auto drelu = [](double x) { return x > 0 ? 1.0 : 0.0; };

    auto sigmoid = [](double x) {
        return 1.0 / (1.0 + std::exp(-x));
    };
    auto dsigmoid = [](double x) {
        double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 - s);
    };
    auto mse = [](double y_pred, double y_true) {
        return 0.5 * std::pow(y_pred - y_true, 2);
    };
    auto dmse = [](double y_pred, double y_true) {
        return y_pred - y_true;
    };

    // Load the MNIST dataset
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    std::cout << "Loading training dataset...\n";

    // Open the training dataset file
    std::ifstream in("../archive/mnist_train.csv");
    if (!in.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir mnist_train.csv" << std::endl;
        return 1;
    }

    // We jump the first line :)
    std::string header;
    std::getline(in, header);

    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ',');
        int label = std::stoi(token);

        // Create a 784x1 matrix for the input (the image)
        // The ith element of the input matrix corresponds to the ith pixel of the image
        Matrix input(784, 1);
        for (int i = 0; i < 784; ++i) {
            // Read each pixel value from the line, convert it to double, and normalize it by dividing by 255.0
            std::getline(ss, token, ',');
            input(i, 0) = std::stod(token) / 255.0;
        }

        // Create a 10x1 matrix for the target (the label)
        // The ith element of the target matrix is 1.0 if the label is i, and 0.0 otherwise
        // This is a one-hot encoding of the label
        Matrix target(10, 1, 0.0);
        target(label, 0) = 1.0;

        inputs.push_back(input);
        targets.push_back(target);
    }
    std::cout << "Training dataset loaded...\n";

    // The next step is to implement a GUI or at least a TUI to let the user choose the parameters of the network
    // For now, we will use a simple network with 3 layers:
    std::vector<std::function<double(double)>> activations = {sigmoid, sigmoid, relu};
    std::vector<std::function<double(double)>> dactivations = {dsigmoid, dsigmoid, drelu};

    // The cost function is the mean squared error (MSE) and its derivative
    Network net({784, 128, 64, 10}, activations, dactivations, mse, dmse, 0.01);
    std::cout << "Building the network with the following parameters : \n";
    std::cout << "Hidden layers :                           128, 24 \n";
    std::cout << "Hidden layers activation function:        sigmoid \n";
    std::cout << "Learning rate:                            0.01 \n";
    std::cout << "Batch size :                              32 \n";
    std::cout << "Epochs :                                  10 \n";

    // Now we have to create the batches
    size_t batch_size = 32;
    int num_epochs = 10;

    std::random_device rd;
    std::mt19937 g(rd());

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Starting " << epoch + 1 << ".\n";

        // Shuffle the indices of the inputs and targets
        // This is done to ensure that the training is not biased by the order of the data
        // We use a random number generator to shuffle the indices
        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        // Split the data into batches
        // We will iterate over the batches and train the network on each batch
        for (size_t b = 0; b < inputs.size() / batch_size; ++b) {
            std::cout << "Epoch " <<  epoch +1 << " on batch " << b+1 << "\n";

            std::vector<Matrix> batch_inputs;
            std::vector<Matrix> batch_targets;
            // We will create a batch of inputs and targets based on the shuffled indices
            // Each batch will contain batch_size elements
            for (size_t i = 0; i < batch_size; ++i) {
                size_t idx = indices[b * batch_size + i];
                batch_inputs.push_back(inputs[idx]);
                batch_targets.push_back(targets[idx]);
            }
            std::cout << "Training... \n";

            // Train the network on the current batch
            net.train(batch_inputs, batch_targets, 1);
            std::cout << "Training finished !\n";

        }

        std::cout << "Epoch " << epoch + 1 << " done.\n";
    }

    // Now we can test the model on the validation data
    std::ifstream test_file("../archive/mnist_test.csv");
    if (!test_file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir mnist_test.csv" << std::endl;
        return 1;
    }

    std::getline(test_file, header);

    std::string test_line;
    int correct = 0;
    int total = 0;
    std::cout << "Running the model on the validation data... \n";

    // We will read the test file line by line, parse the input and label, and then use the network to predict the label
    // The input is a 784x1 matrix (the image) and the output is a 10x1 matrix (the predicted label)
    // We will compare the predicted label with the actual label and count the number of correct predictions
    while (std::getline(test_file, test_line)) {
        std::stringstream ss(test_line);
        std::string token;

        std::getline(ss, token, ',');
        int label = std::stoi(token);

        Matrix input(784, 1);
        for (int i = 0; i < 784; ++i) {
            std::getline(ss, token, ',');
            input(i, 0) = std::stod(token) / 255.0;
        }

        Matrix output = net.forward(input);

        // Find the index of the maximum value in the output matrix
        int predicted = 0;
        double max_val = output(0, 0);
        for (int i = 1; i < 10; ++i) {
            if (output(i, 0) > max_val) {
                max_val = output(i, 0);
                predicted = i;
            }
        }

        if (predicted == label) correct++;
        total++;
    }
    std::cout << "Validation completed ! \n";

    double accuracy = static_cast<double>(correct) / total * 100.0;
    std::cout << "Model accuracy: " << accuracy << "%\n";

    return 0;
}
