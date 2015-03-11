//
//  MLPNeuralNet.m
//  BIOMotionTracker
//
//  Created by Mykola Pavlov on 9/23/13.
//  Copyright (c) 2013 Biomech, Inc. All rights reserved.
//

#import "MLPNeuralNet.h"

#define BIAS_VALUE 1.0
#define BIAS_UNIT 1

typedef struct {
    NSInteger nrow;      // Number of rows
    NSInteger ncol;      // Number of columns
    double *weightMatrix;
} MLPLayer;

@interface MLPNeuralNet () {
    NSMutableData *hiddenFeatures;
    NSMutableData *buffer;
    NSData *arrayOfLayers; // MLPLayer structures
}
@end

@implementation MLPNeuralNet

#pragma mark - Initializer and dealloc

// Designated initializer
- (id)initWithLayerConfig:(NSArray *)layerConfig
                  weights:(NSData *)weights
               outputMode:(MLPOutput)outputMode {
    self = [super init];
    if (self) {
        if ([self.class countWeights:layerConfig] != weights.length / sizeof(double)) {
            @throw [NSException exceptionWithName:NSInternalInconsistencyException
                                           reason:@"Number of weights doesn't match to configuration"
                                         userInfo:nil];
        }
        
        _numberOfLayers = layerConfig.count;
        _featureVectorSize = [layerConfig[0] unsignedIntegerValue] * sizeof(double);
        _predictionVectorSize = [layerConfig.lastObject unsignedIntegerValue] * sizeof(double);
        _outputMode = outputMode;
        
        // Allocate buffers of the maximum possible vector size, there should be a place for bias unit also.
        unsigned maxVectorLength = [[layerConfig valueForKeyPath:@"@max.self"] unsignedIntValue] + BIAS_UNIT;
        hiddenFeatures = [NSMutableData dataWithLength:maxVectorLength * sizeof(double)];
        buffer = [NSMutableData dataWithLength:maxVectorLength * sizeof(double)];
        
        // Allocate memory for layers. Note that we don't need a matrix for
        // the input layer, so the total size is equal to number of layers - 1.
        arrayOfLayers = [NSMutableData dataWithLength:(_numberOfLayers - 1) * sizeof(MLPLayer)];
        
        // Allocate memory for the wigth matrices and initialize them.
        MLPLayer *layer = (MLPLayer *)arrayOfLayers.bytes;
        double *wts = (double *)weights.bytes;
        int crossLayerOffset = 0; // An offset between the weight matrices of different layers
        for (int j = 0; j < _numberOfLayers - 1; j++) { // Recall we don't need a matrix for the input layer
            
            // If network has X units in layer j, and Y units in layer j+1, then weight matrix for layer j
            // will be of demension: [ Y x (X+1) ]
            layer[j].nrow = [layerConfig[j+1] unsignedIntegerValue];
            layer[j].ncol = [layerConfig[j] unsignedIntegerValue] + BIAS_UNIT;
            layer[j].weightMatrix = calloc(layer[j].nrow * layer[j].ncol, sizeof(double));
            NSAssert(layer[j].weightMatrix != NULL, @"Out of memory for weight matrices");
            
            int totalOffset = 0;
            for (int row = 0; row < layer[j].nrow; row++) {
                for (int col = 0; col < layer[j].ncol; col++) {
                    int crossRowOffset = (col + row * (int)layer[j].ncol); // Simulate the matrix using row-major ordering
                    totalOffset = crossRowOffset + crossLayerOffset;  // Now matrix[offset] corresponds to M[row, col]
                    layer[j].weightMatrix[crossRowOffset] = wts[totalOffset];
                }
            }
            crossLayerOffset = totalOffset + 1; // Adjust offset to the next layer
        }
        
    }
    return self;
}

- (id)init {
    @throw [NSException exceptionWithName:@"MLPNeuralNet init"
                                   reason:@"Use designated initializer, not init"
                                 userInfo:nil];
}

- (void)dealloc {
    // Free weight matrices in each layer
    MLPLayer *layer = (MLPLayer *)arrayOfLayers.bytes;
    for (int j = 0; j < self.numberOfLayers - 1; j++) {
        free(layer[j].weightMatrix); layer[j].weightMatrix = NULL;
    }
}

#pragma mark - Prediction

- (void)predictByFeatureVector:(NSData *)vector intoPredictionVector:(NSMutableData *)prediction {
    if (vector.length < self.featureVectorSize) {
        @throw [NSException exceptionWithName:NSInternalInconsistencyException
                                       reason:@"Feature-vector size is less than specified in configuration"
                                     userInfo:nil];
    } else if (prediction.length < self.predictionVectorSize) {
        @throw [NSException exceptionWithName:NSInternalInconsistencyException
                                       reason:@"Prediction vector size is less than specified in configuration"
                                     userInfo:nil];
    }
    
    // Copy feature-vector into buffer and add the bias unit at index 0
    double *features = (double *)hiddenFeatures.mutableBytes;
    features[0] = BIAS_VALUE;
    memcpy(&features[1], (double *)vector.bytes, self.featureVectorSize);
    
    //
    // Forward propagation algorithm
    //
    double *buf = (double *)buffer.mutableBytes;
    MLPLayer *layer = (MLPLayer *)arrayOfLayers.bytes;
    for (int j = 0; j < self.numberOfLayers - 1; j++) {
        
        // 1. Calculate hidden features for current layer j
        vDSP_mmulD(layer[j].weightMatrix, 1, features, 1, &buf[1], 1, layer[j].nrow, 1, layer[j].ncol);
        
        // 2. Add the bias unit at index 0 and propagate features to the next level
        buf[0] = BIAS_VALUE;
        memcpy(features, buf, (layer[j].nrow + BIAS_UNIT) * sizeof(double));
        
        // 3. Apply logistic activation function if needed: http://en.wikipedia.org/wiki/Logistic_function
        if (self.outputMode == MLPClassification) {
            for (int i = 0; i < layer[j].nrow; i++) { // Can we use Taylor's theorem to vectorize this loop?
                features[i + BIAS_UNIT] = 1 / (1 + exp(-features[i + BIAS_UNIT])); // But skip the bias unit
            }
        }
    }
    // 4. Copy an assessment into prediction vector
    memcpy((double *)prediction.mutableBytes, &features[1], self.predictionVectorSize);
}

#pragma mark - Misc

- (NSString *)description {
    MLPLayer *layer = (MLPLayer *)arrayOfLayers.bytes;
    NSMutableString *networkArch = [NSMutableString string];
    int numberOfWeights = 0;
    for (int i = 0; i < arrayOfLayers.length / sizeof(MLPLayer); i++) {
        numberOfWeights += layer[i].ncol * layer[i].nrow;
        [networkArch appendFormat:@"%d-", (int)layer[i].ncol - 1];
    }
    [networkArch appendFormat:@"%lu", self.predictionVectorSize / sizeof(double)];
    return [NSString stringWithFormat:@"a %@ network with %d weigths", networkArch, numberOfWeights];
}

+ (NSInteger)countWeights:(NSArray *)layerConfig {
    NSInteger count = 0;
    for (int i = 0; i < layerConfig.count - 1; i++) {
        count += ([layerConfig[i] unsignedIntValue] + 1) * [layerConfig[i+1] unsignedIntValue];
    }
    return count;
}



@end
