#!/bin/bash
echo "Compiling Neural Network"
rm -rf nnet
g++ -o nnet nnet.cpp -std=c++17
echo "Finished Compiling"
exit 0
