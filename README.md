# dumbrons.

## What is this?

`dumbrons` is a simple from-scratch implementation of a **feedforward neural network** in **modern C++**.  
It is built as a learning project and is intended to demonstrate how to:

- Load and preprocess real datasets (like MNIST in CSV format)
- Design a modular neural network architecture (layers, forward/backward passes)
- Train using batch-based gradient descent
- Use different activation and loss functions

⚠️ This project prioritizes **clarity over performance**, and avoids external ML libraries (like TensorFlow or PyTorch).

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dumbrons.git
cd dumbrons
```
### 2. Build the project
```bash
cd build
cmake ..
make
```
### 3. Run the training
Place the MNIST CSV files (mnist_train.csv and mnist_test.csv) in an archive/ directory at the root.

```bash
./dumbrons
```
### Customization
Edit main.cpp to change Network architecture, activation functions (ReLU, Sigmoid), loss function (MSE), number of epochs, batch size, learning rate

A future improvement would be to allow full configuration through a command-line interface or JSON config file.


## Project structure
```
.
├── archive/           # Dataset files (CSV)
├── build/             # Compiled binaries
├── CMakeLists.txt     # Build configuration
├── main.cpp           # Training loop
├── matrix.*           # Matrix implementation
├── layer.*            # Layer structure
├── network.*          # Neural network class
├── roadmap.md         # TODOs and ideas
└── testmatrix.cpp     # Matrix unit tests
```
### Dataset
This project uses the MNIST dataset converted to CSV format.
mnist_train.csv – for training
mnist_test.csv – for evaluation
Make sure both are in the archive/ folder.

### License
This project is open-source and licensed under the MIT License.
Feel free to fork, modify, or contribute!

Made with ❤️ and a lot of `std::cout` :)
