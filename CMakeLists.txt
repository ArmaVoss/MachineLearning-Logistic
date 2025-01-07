cmake_minimum_required(VERSION 3.30)
project(MachineLearning)

set(CMAKE_CXX_STANDARD 20)

add_executable(MachineLearning main.cpp
        Vec.cpp
        Vec.h
        Matrix.cpp
        Matrix.h
        Parameter.h
)
