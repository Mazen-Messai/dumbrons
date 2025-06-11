cmake_minimum_required(VERSION 3.31)
project(dumbrons)

set(CMAKE_CXX_STANDARD 20)

add_executable(testmatrix testmatrix.cpp matrix.cpp
        layer.cpp
        layer.h
        network.cpp
        network.h)

add_executable(dumbrons main.cpp matrix.cpp
        layer.cpp
        layer.h
        network.cpp
        network.h)