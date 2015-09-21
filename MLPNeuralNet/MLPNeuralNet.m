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
#define ReLU_THR 0.0

typedef struct {
    // Number of rows
    NSInteger nrow;
    // Number of columns
    NSInteger ncol;
    double *weightMatrix;
} MLPLayer;

@interface MLPNeuralNet () {
    
    NSMutableData *hiddenFeatures;
    NSMutableData *buffer;
    // MLPLayer structures
    NSData *arrayOfLayers;
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
                    // Simulate the matrix using row-major ordering
                    int crossRowOffset = (col + row * (int)layer[j].ncol);
                    // Now matrix[offset] corresponds to M[row, col]
                    totalOffset = crossRowOffset + crossLayerOffset;
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
    [self predictByFeatureMatrix:vector intoPredictionMatrix:prediction];
}

- (void)predictByFeatureMatrix:(NSData *)matrix intoPredictionMatrix:(NSMutableData *)prediction {
    // Number of examples we are going to classify
    long numExamples = matrix.length / self.featureVectorSize;
    long numFeatures = self.featureVectorSize / sizeof(double);
    
    if (matrix.length != self.featureVectorSize * numExamples) {
        NSString* error = [NSString stringWithFormat:@"Size of feature matrix invalid (in bytes). Got: %lx Expected: %lx",
                                      (unsigned long)matrix.length, self.featureVectorSize * numExamples];
        @throw [NSException exceptionWithName:NSInternalInconsistencyException reason:error userInfo:nil];
    } else if (prediction.length != self.predictionVectorSize * numExamples) {
        NSString* error = [NSString stringWithFormat:@"Size of prediction matrix invalid (in bytes). Got: %lx Expected: %lx",
                           (unsigned long)prediction.length, self.predictionVectorSize * numExamples];
        @throw [NSException exceptionWithName:NSInternalInconsistencyException reason:error userInfo:nil];
    }
    
    MLPLayer *layer = (MLPLayer *)arrayOfLayers.bytes;
    double bias = BIAS_VALUE;
    
    double *features;
    double *buf;

    // Create buffers, reusing buffers where possible
    if(numExamples == 1){
        features = (double *)hiddenFeatures.mutableBytes;
        buf = (double *)buffer.mutableBytes;
    } else {
        NSMutableData *hiddenFeatureM = [NSMutableData dataWithLength:hiddenFeatures.length * numExamples];
        NSMutableData *bufferM = [NSMutableData dataWithLength:hiddenFeatures.length * numExamples];
        
        features = (double *)hiddenFeatureM.mutableBytes;
        buf = (double *)bufferM.mutableBytes;
    }
    
    // Add the bias unit in the first row
    vDSP_vfillD(&bias, &features[0], 1, numExamples);
    
    // Copy feature-matrix into the buffer. We will transpose the feature matrix to get the
    // bias units in a row instead of a column for easier updates.
    vDSP_mtransD((double *)matrix.bytes, 1, &features[numExamples], 1, numFeatures, numExamples);
    
    // Forward propagation algorithm
    for (int j = 0; j < self.numberOfLayers - 1; j++) {
        
        // 1. Calculate hidden features for current layer j
        vDSP_mmulD(layer[j].weightMatrix, 1, features, 1, &buf[numExamples], 1, layer[j].nrow, numExamples, layer[j].ncol);
        
        // 2. Add the bias unit in row 0 and propagate features to the next level
        vDSP_vfillD(&bias, &buf[0], 1, numExamples);
        
        memcpy(features, buf, (layer[j].nrow + BIAS_UNIT) * numExamples * sizeof(double));
        
        // 3. Apply activation function, e.g. logistic func: http://en.wikipedia.org/wiki/Logistic_function
        if (self.outputMode == MLPClassification) {
            int feature_len = (int)(layer[j].nrow * numExamples);
            double one = 1.0;
            double mone = -1.0;
            double relu_threshold = ReLU_THR;
            
            MLPActivationFunction activation =
            (j < self.numberOfLayers - 2) ? self.hiddenActivationFunction : self.outputActivationFunction;
            
            switch (activation) {
                case MLPSigmoid:
                    vDSP_vnegD(&features[numExamples], 1, &features[numExamples], 1, feature_len);
                    vvexp(&features[numExamples], &features[numExamples], &feature_len);
                    vDSP_vsaddD(&features[numExamples], 1, &one, &features[numExamples], 1, feature_len);
                    vvpows(&features[numExamples], &mone, &features[numExamples], &feature_len);
                    break;
                    
                case MLPTangent:
                    vvtanh(&features[numExamples], &features[numExamples], &feature_len);
                    break;
                    
                case MLPReLU:
                    vDSP_vthresD(&features[numExamples], 1, &relu_threshold, &features[numExamples], 1, feature_len);
                    break;
                case MLPNone:
                    break;
            }
        }
    }
    
    // 4. Copy results into prediction matrix
    memcpy((double *)prediction.mutableBytes, &features[numExamples], self.predictionVectorSize * numExamples);
}

#pragma mark - Activation Function

- (MLPActivationFunction)activationFunction {
    return [self hiddenActivationFunction];
}

- (void)setActivationFunction: (MLPActivationFunction)activation {
    _hiddenActivationFunction = activation;
    _outputActivationFunction = activation;
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
