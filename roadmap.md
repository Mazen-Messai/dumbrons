# Neural Network from Scratch in C — Roadmap

## Goal
- Forward propagation
- Backpropagation
- Activation functions (ReLU, Sigmoid)
- Loss functions (MSE or Cross-Entropy)
- Optimization (SGD, optionally Adam)
- Batch-based training
- No external libraries

---

## Day 1 – Data Structures & Math Utilities

- [x] Define a `Matrix` structure (float**, rows, cols)
- [x] Define a `Layer` structure (weights, biases, activation, output, gradients)
- [ ] Define a `Network` structure (array of layers)
- [x] Implement basic matrix operations:
    - [x] Matrix multiplication (`matrix_dot`)
    - [x] Matrix addition/subtraction
    - [x] Matrix transpose
    - [ ] Element-wise function application
- [ ] Add random matrix initialization
- [ ] Write simple unit tests for all matrix ops

---

## Day 2 – Forward Propagation

- [ ] Implement affine transformation: `Z = W * X + b`
- [ ] Apply activation functions (ReLU, Sigmoid)
- [ ] Create a `forward(Network*, Matrix* input)` function
- [ ] Print output on test inputs

---

## Day 3 – Backpropagation

- [ ] Compute derivative of loss (dL/dY)
- [ ] Implement derivative of activation functions
- [ ] Chain gradients backward through the network
- [ ] Accumulate weight and bias gradients
- [ ] Create a `backward(Network*, Matrix* target)` function

---

## Day 4 – Training Loop

- [ ] Build `train(Network*, inputs, targets)` function
- [ ] Include loop over epochs
- [ ] Run: forward → backward → update weights
- [ ] Implement Mean Squared Error (MSE) loss
- [ ] Optionally: Cross-Entropy loss

---

## Day 5 – Optimization

- [ ] Implement vanilla Stochastic Gradient Descent (SGD)
- [ ] Add learning rate as a parameter
- [ ] Implement momentum (optional)
- [ ] Implement Adam optimizer (optional)
- [ ] Support mini-batch gradient descent

---

## Day 6 – Data Loading & Evaluation

- [ ] Hardcode XOR dataset or use synthetic blobs
- [ ] Implement simple CSV loader
- [ ] Train on multiple datasets
- [ ] Add accuracy evaluation after training

---

## Day 7 – Cleanup and Presentation

- [ ] Refactor code: modular headers and functions
- [ ] Write a clear and minimal `README.md`:
    - [ ] Project overview
    - [ ] Compilation instructions
    - [ ] Architecture explanation
- [ ] Log loss values over epochs
- [ ] Optionally: script to plot loss curve (e.g. with Python or gnuplot)
- [ ] Prepare showcase materials (GIFs, screenshots, demo output)

---

## Bonus Ideas

- [ ] Add a CLI tool to classify inputs manually
- [ ] Visualize decision boundaries (export a .ppm image)
- [ ] Export weights to file for later reuse
- [ ] Add command-line flags (learning rate, epochs, batch size)
