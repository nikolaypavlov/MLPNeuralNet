[![Build Status](https://travis-ci.org/nikolaypavlov/neural-net-for-iOS.png?branch=master)](https://travis-ci.org/nikolaypavlov/neural-net-for-iOS)
# MLPNeuralNet
Fast [multilayer perceptron](http://en.wikipedia.org/wiki/Multilayer_perceptron) neural network library for iOS and Mac OS X. It is built on top of the [Apple's Accelerate Framework](https://developer.apple.com/library/ios/documentation/Accelerate/Reference/AccelerateFWRef/_index.html), using vectorized operations and hardware acceleration if available.

## Why to choose it?
Imagine that you created a prediction model in Matlab (Python or R) and want to use it in iOS app. If that's the case, MLPNeuralNet is exactly what you need. It is specifically designed to load and run models in [forward propagation](http://en.wikipedia.org/wiki/Backpropagation#Phase_1:_Propagation) mode only.

### Features:
- [classification](http://en.wikipedia.org/wiki/Binary_classification), [multiclass classification](http://en.wikipedia.org/wiki/Multiclass_classification) and regression output;
- vectorized implementaion;
- works with double precision;
- multiple hidden layers or none (in that case it's same as logistic/linear regression)

## Quick Example
Let's deploy a model for the AND function ([conjunction](http://en.wikipedia.org/wiki/Logical_conjunction)) that works as follows: 

|X1 |X2 | Y |
|:-:|:-:|:-:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

Our model has the following weights and network configuration:


```objectivec

// Use designated initializer to pass configuration and weights to the model
MLPNeuralNet *model = [[MLPNeuralNet alloc] initWithLayerConfig:@[@2, @1] 
                                                        weights:@[@-30, @20, @20] 
                                                     outputMode:MLPClassification];

// Predict output of the model for data sample
double sample[] = {0, 1};
vector = [NSData dataWithBytes:sample length:sizeof(sample)];
prediction = [NSMutableData dataWithLength:sizeof(double)];
[model predictByFeatureVector:vector intoPredictionVector:prediction];

double assessment = (double *)prediction.bytes;
NSLog(@"Model assessment is %f", assessment[0]);

```

## Unit Tests
MLPNeuralNet includes a suite of unit tests in the MLPNeuralNetTests subdirectory. You can execute them via the "MLPNeuralNet" scheme within Xcode.

## Credits
MLPNeuralNet implementation was inspired by [Andrew Ng's Machine Learning course](https://www.coursera.org/course/ml) on Coursera.
Artificial Neural Net image was taken from [Wikipedia Commons](http://en.wikipedia.org/wiki/File:Artificial_neural_network.svg)

## Contact

Maintainer: [Mykola Pavlov](http://github.com/nikolaypavlov/) (me@nikolaypavlov.com)

## License
MLPNeuralNet is available under the BSD license. See the LICENSE file for more info.

