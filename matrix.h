//
// Created by Mazen Messai on 08/06/2025.
// This file is part of a simple matrix library for C++.
// It provides basic matrix operations such as addition, multiplication, and transposition.
// The library is designed to be efficient and easy to use, with a focus on numerical computations.
// It has been written to be used in machine learning applications, particularly for neural networks.
// It is just a simple implementation of a matrix class, not optimized for performance or memory usage.
// I wrote it to practice C++ and have fun with machine learning concepts.
//
// This file is released under the MIT License.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cassert>

class Matrix {
private:
    // The matrix data is stored in a flat vector for efficiency
    // Maybe i will implementing hollow matrices in the future
    std::vector<double> data;
    size_t rows;
    size_t cols;

public:
    // Constructor to create a matrix of given size initialized with a specific valu
    //
    // Parameters :
    // rows : number of rows in the matrix
    // cols : number of columns in the matrix
    // init_val : initial value for all elements in the matrix, default is 0.0
    Matrix(size_t rows, size_t cols, double init_val = 0.0);

    size_t numRows() const { return rows; }
    size_t numCols() const { return cols; }

    // Access elements using (row, column) indexing
    //
    // Parameters :
    // i : row index
    // j : column index
    // output : a reference to the element at (i, j)
    //
    // Throws std::out_of_range if the indices are out of bounds
    double& operator()(size_t i, size_t j);
    double operator()(size_t i, size_t j) const;

    void print(std::ostream& out = std::cout, int precision = 4) const;

    // Returns a row as a new matrix
    //
    // Parameters :
    // i : row index
    // out : a new matrix containing the specified row, should be size (1, cols)
    //
    // Throws std::out_of_range if the row index is out of bounds
    Matrix row(size_t i) const;

    // Returns a column as a new matrix
    //
    // Parameters :
    // j : column index
    // Returns a new matrix containing the specified column, should be size (rows, 1)
    //
    // Throws std::out_of_range if the column index is out of bounds
    Matrix col(size_t j) const;

    // Returns the transpose of the matrix
    // output : a new matrix that is the transpose of the current matrix, should be size (cols, rows)
    Matrix transpose() const;

    // Creates an identity matrix of size n x n
    //
    // Parameters :
    // n : size of the identity matrix
    //
    // Returns a new identity matrix of size (n, n)
    static Matrix identity(size_t n);

    // Matrix addition and multiplication operators
    //
    // Parameters :
    // other : the other matrix to add or multiply with
    // output : a new matrix that is the result of the operation
    //
    // Adds this matrix to another matrix
    Matrix operator+(const Matrix& other) const;

    // Multiplies this matrix by another matrix
    //
    // Parameters :
    // other : the other matrix to multiply with
    // output : a new matrix that is the result of the multiplication
    //
    // Throws std::invalid_argument if the matrices cannot be multiplied (i.e., if the number of columns in this matrix does not match the number of rows in the other matrix)
    Matrix operator*(const Matrix& other) const;

    //Matrix parallelMultiply(const Matrix& other) const;

    // Adds another matrix to this matrix in place
    //
    // Parameters :
    // other : the other matrix to add
    // output : a reference to this matrix after the addition
    //
    // Throws std::invalid_argument if the matrices do not have the same dimensions
    Matrix& operator+=(const Matrix& other);
};

#endif //MATRIX_H
