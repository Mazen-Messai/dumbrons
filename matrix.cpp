//
// Created by Mazen Messai on 08/06/2025.
//
// This file is released under the MIT License.
//

#include "matrix.h"

Matrix::Matrix(size_t rows, size_t cols, double init_val)
    : rows(rows), cols(cols), data(rows * cols, init_val)
{
}

double& Matrix::operator()(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Index out of bounds");
    }
    // Accessing the element at (i, j) in a 1D vector representation
    // The element at (i, j) is at index i * cols + j in the data vector
    // This is a common way to store 2D matrices in a 1D array for performance
    return data[i * cols + j];
}

double Matrix::operator()(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data[i * cols + j];
}

void Matrix::print(std::ostream& out, int precision) const {
    // The printing on this function is very ugly but it is just for debugging purposes
    out << std::fixed << std::setprecision(precision);
    for (size_t i = 0; i < rows; ++i) {
        out << "[ ";
        for (size_t j = 0; j < cols; ++j) {
            out << (*this)(i, j);
            if (j < cols - 1) out << ", ";
        }
        out << " ]\n";
    }
}

Matrix Matrix::row(size_t i) const {
    if (i >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }

    Matrix r(1, cols);
    for (size_t j = 0; j < cols; ++j) {
        r(0, j) = (*this)(i, j);
    }

    return r;
}

Matrix Matrix::col(size_t j) const {
    if (j >= cols) {
        throw std::out_of_range("Column index out of bounds");
    }

    Matrix c(rows, 1);
    for (size_t i = 0; i < rows; ++i) {
        c(i, 0) = (*this)(i, j);
    }

    return c;
}

Matrix Matrix::transpose() const {
    Matrix m(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(j, i) = (*this)(i, j);
        }
    }

    return m;
}

Matrix Matrix::identity(size_t n) {
    Matrix m(n, n, 0);

    for (size_t i = 0; i < n; ++i) {
        m(i, i) = 1.0;
    }

    return m;
}

Matrix Matrix::operator+(const Matrix& b) const {
    if (rows != b.rows || cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix m(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = (*this)(i, j) + b(i, j);
        }
    }

    return m;
}

Matrix Matrix::operator*(const Matrix& b) const {
    if (cols != b.rows) {
        throw std::invalid_argument("In order to perform A • B, cols(A) must match rows(B)");
    }

    Matrix m(rows, b.cols, 0.0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < b.cols; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                m(i, j) += (*this)(i, k) * b(k, j);
            }
        }
    }

    return m;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            (*this)(i, j) += other(i, j);
        }
    }

    return *this;
}

