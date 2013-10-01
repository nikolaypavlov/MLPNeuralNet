[![Build Status](https://travis-ci.org/nikolaypavlov/neural-net-for-iOS.png?branch=master)](https://travis-ci.org/nikolaypavlov/neural-net-for-iOS)
# MLPNeuralNet
Fast multilayer perceptron neural network library for iOS and Mac OS X. It is built on top of the [Apple's Accelerate Framework](https://developer.apple.com/library/ios/documentation/Accelerate/Reference/AccelerateFWRef/_index.html), using vectorized operations and hardware acceleration if available.

## Why to use it?
Imagine that you've created a prediction model in Matlab (Python or R) and you want to deploy it on iOS. If that's the case, MLPNeuralNet is a great choise for you. It is specifically designed to load and run models in a forward propagation mode only.
