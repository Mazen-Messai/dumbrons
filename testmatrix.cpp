//
// Created by mazen on 08/06/2025.
//

#include <iostream>
#include <cassert>
#include "matrix.h"

int main() {
    printf("============ Starting Tests ============\n");
    Matrix A(2, 3, 1.0);
    A(0, 1) = 5.0;
    assert(A(0, 1) == 5.0);
    assert(A(1, 2) == 1.0);
    printf("Test 1 passed.\n");

    Matrix r = A.row(1);
    assert(r.numRows() == 1 && r.numCols() == 3);
    assert(r(0, 2) == A(1, 2));
    printf("Test 2 passed.\n");

    Matrix c = A.col(1);
    assert(c.numRows() == 2 && c.numCols() == 1);
    assert(c(0, 0) == A(0, 1));
    printf("Test 3 passed.\n");

    Matrix At = A.transpose();
    assert(At.numRows() == A.numCols());
    assert(At.numCols() == A.numRows());
    assert(At(1, 0) == A(0, 1));  // transposé
    printf("Test 4 passed.\n");

    Matrix I = Matrix::identity(3);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) assert(I(i, j) == 1.0);
            else assert(I(i, j) == 0.0);
        }
    }
    printf("Test 5 passed.\n");

    Matrix B(2, 3, 2.0);
    Matrix C = A + B;
    assert(C(0, 0) == 3.0);
    assert(C(0, 1) == 7.0); // 5.0 + 2.0
    assert(C(1, 2) == 3.0);
    printf("Test 6 passed.\n");

    // Test 6: +=
    A += B;
    assert(A(0, 1) == 7.0);
    assert(A(1, 0) == 3.0);
    printf("Test 7 passed.\n");

    Matrix D(3, 2, 1.0);
    Matrix E = B * D; // B: 2x3, D: 3x2 → E: 2x2
    assert(E.numRows() == 2 && E.numCols() == 2);
    assert(E(0, 0) == 6.0);
    printf("Test 8 passed.\n");

    printf("================ Success ===============");
    return 0;
}
