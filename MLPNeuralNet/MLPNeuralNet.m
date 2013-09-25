//
//  MLPNeuralNet.m
//  BIOMotionTracker
//
//  Created by Mykola Pavlov on 9/23/13.
//  Copyright (c) 2013 Biomech, Inc. All rights reserved.
//

#import "MLPNeuralNet.h"

#define BIAS 1.0

typedef struct {
    NSUInteger nrow; // Number of rows
    NSUInteger ncol; // Number of columns
    double *weightMatrix;
} MLPWeightMatrix;

@interface MLPNeuralNet () {
    NSMutableData *featureVector;
    NSMutableData *outputBuffer;
    MLPWeightMatrix *layer; // Array of weight matrices for each layer
}

@property (readonly, nonatomic) MLPOutput outputMode;
@property (readonly, nonatomic) NSArray *neuronsInLayer;
@property (readonly, nonatomic) NSUInteger numberOfLayers;

@end

@implementation MLPNeuralNet

#pragma mark - Initializer and dealloc

// Designated initializer
- (id)initWithLayersConfig:(NSArray *)layersConfig
                   weights:(NSArray *)weights
                outputMode:(MLPOutput)outputMode {
    self = [super init];
    if (self) {
        for (int i = 0; i < layersConfig.count; i ++) {
            if (![[layersConfig objectAtIndex:i] isKindOfClass:[NSNumber class]]) {
                @throw [NSException exceptionWithName:@"MLPNeuralNet initializer"
                                               reason:@"Network configuraion should be a collection of NSNumbers"
                                             userInfo:nil];
            }
        }
        
        for (int i = 0; i < weights.count; i++) {
            if (![[weights objectAtIndex:i] isKindOfClass:[NSNumber class]]) {
                @throw [NSException exceptionWithName:@"MLPNeuralNet initializer"
                                               reason:@"Weight matrices should be a collection of NSNumbers"
                                             userInfo:nil];
            }
        }
        
        _numberOfLayers = layersConfig.count;
        _neuronsInLayer = layersConfig;
        _outputMode = outputMode;
        
        // Allocate buffers for the maximum possible vector size, there should be a place for bias unit also.
        unsigned maxVectorLength = [[layersConfig valueForKeyPath:@"@max.self"] unsignedIntValue] + 1;
        featureVector = [NSMutableData dataWithLength:maxVectorLength * sizeof(double)];
        outputBuffer = [NSMutableData dataWithLength:maxVectorLength * sizeof(double)];
        
        // Allocate memory for the array of weight matrices. Note that we don't need a matrix for
        // the input layer, so the total number is equal to numberOfLayers - 1.
        layer = calloc(_numberOfLayers - 1, sizeof(MLPWeightMatrix));
        NSAssert(layer != NULL, @"Out of memory for layers");
        
        // Let's allocate resources for the wigth matrices and initialize them.
        // If network has X units in layer j, Y units in layer j+1, then weight matrix
        // for layer j will be of demension: Y x (X+1).
        for (int j = 0; j < _numberOfLayers - 1; j++) { // Recall we don't need a matrix for the input layer
            layer[j].nrow = [[_neuronsInLayer objectAtIndex:j] unsignedIntegerValue];
            layer[j].ncol = [[_neuronsInLayer objectAtIndex:j + 1] unsignedIntegerValue] + 1;
            
            // Allocate memory for the weight matrix of current layer.
            layer[j].weightMatrix = calloc(layer[j].nrow * layer[j].ncol, sizeof(double));
            NSAssert(layer[j].weightMatrix != NULL, @"Out of memory for weight matrices");
            
            // Now let's initialize weigths
            for (int row = 0; row < layer[j].nrow; row++) {
                for (int col = 0; col < layer[j].ncol; col++) {
                    // Simulate the matrix using row-major ordering. Now matrix[offset] corresponds to M[row, col]
                    int offset = row * layer[j].ncol + col;
                    layer[j].weightMatrix[offset] = [[weights objectAtIndex:offset] doubleValue];
                }
            }
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
    for (int j = 0; j < self.numberOfLayers - 1; j++) {
        free(layer[j].weightMatrix); layer[j].weightMatrix = NULL;
    }
    // Then free layer itself
    free(layer); layer = NULL;
}

#pragma mark - Prediction

- (NSNumber *)predictByFeatureVector:(NSArray *)vector {
    [self replaceBuffer:featureVector withBiasAndVector:vector];
    double *features = (double *)featureVector.mutableBytes;
    double *outputVector = (double *)outputBuffer.mutableBytes;
    
    // Forward propagation algorithm
    for (int j = 0; j < self.numberOfLayers - 1; j++) {
        // Calculate feature-vector for current layer j.
        vDSP_mmulD(layer[j].weightMatrix, 1, features, 1, &outputVector[1], 1, layer[j].nrow, 1, layer[j].ncol);
        
        // Insert bias unit at index 0 and overwrite old feature-vector with the new one
        outputVector[0] = BIAS;
        memcpy(features, outputVector, layer[j].nrow * sizeof(double));
        
        // Apply logistic activation function if needed: http://en.wikipedia.org/wiki/Logistic_function
        if (self.outputMode == MLPClassification) {
            for (int i = 1; i < layer[j].nrow; i++) { // Skip bias unit
                features[i] = 1 / (1 + exp(-features[i])); // Maybe Taylor's theorem can be used to vectorize this?
            }
        }
        // Propagate to the next level...
    }
    return [NSNumber numberWithDouble:features[1]];
}

#pragma mark - Misc

// Copies content of the NSArray into C array and adds bias unit at index 0
- (void)replaceBuffer:(NSMutableData *)buffer withBiasAndVector:(NSArray *)vector {
    NSAssert(vector.count <= buffer.length / sizeof(double), @"Input vector size exceeds the maximum vector in configuration");
    
    double *features = (double *)buffer.mutableBytes;
    features[0] = BIAS;
    for (int i = 1; i < vector.count + 1; i++) {
        features[i] = [[vector objectAtIndex:i] doubleValue];
    }

}

@end
