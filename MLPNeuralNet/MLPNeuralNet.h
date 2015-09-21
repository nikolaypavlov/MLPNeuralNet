//
//  MLPNeuralNet.h
//  BIOMotionTracker
//
//  Created by Mykola Pavlov on 9/23/13.
//  Copyright (c) 2013 Biomech, Inc. All rights reserved.
//
//  **********************************************************************************
//
//  This library is designed to predicts new examples by trained neural net.
//  Use it for regression, classification and multiclass classification problems.
//
//  MLPNeuralNet is fully vectorized, it uses double precision to store the data and
//  you can use as many hidden layers as you want.
//
//  NOTE: if your feature-vector containes class/factor variables, it's your duty
//  to provide them in portable way. For example, if you encode "red", "green", "blue"
//  as integers 1, 2, 3 in Matlab, you should use the same encoding for MLPNeuralNet.
//
//  **********************************************************************************

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>

// Type of output. The logistic activitation function is used for classification:
// http://en.wikipedia.org/wiki/Logistic_function
typedef enum {
    // Linear output from -Inf to +Inf
    MLPRegression,
    // Interval from 0 to 1 for sigmoid and -1 to 1 for tangent
    MLPClassification,
} MLPOutput;

typedef enum {
    // Sigmoid is the default
    MLPSigmoid,
    MLPTangent,
    MLPReLU, // Rectified Linear Unit, it computes point-wise y=max(0,x)
    MLPNone,
} MLPActivationFunction;

@interface MLPNeuralNet : NSObject

@property (readonly, nonatomic) NSUInteger numberOfLayers;
@property (readonly, nonatomic) NSUInteger featureVectorSize; // in bytes
@property (readonly, nonatomic) NSUInteger predictionVectorSize; // in bytes
@property (readonly, nonatomic) MLPOutput outputMode;
@property (nonatomic) MLPActivationFunction hiddenActivationFunction;
@property (nonatomic) MLPActivationFunction outputActivationFunction;
@property (nonatomic) MLPActivationFunction activationFunction;

// Designated initializer
- (id)initWithLayerConfig:(NSArray *)layerConfig // of NSNumbers
                  weights:(NSData *)weights      // of double
               outputMode:(MLPOutput)outputMode;

// Predicts new examples by feature-vector and copies the prediction into specified buffer
// Vector and prediction buffers should be allocated to work with double precision
- (void)predictByFeatureVector:(NSData *)vector intoPredictionVector:(NSMutableData *)prediction;

// Predicts multiple new examples by feature-matrix and copies the predictions into specified
// prediction matrix. The predictions buffer should be allocated to work with double precision.
- (void)predictByFeatureMatrix:(NSData *)matrix intoPredictionMatrix:(NSMutableData *)prediction;

// Number of weights requred for the neural net of this configuration
+ (NSInteger)countWeights:(NSArray *)layerConfig;

@end
