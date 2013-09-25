//
//  MLPNeuralNet.h
//  BIOMotionTracker
//
//  Created by Mykola Pavlov on 9/23/13.
//  Copyright (c) 2013 Biomech, Inc. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

// Type of output. The logistic activitation is used for classification problems:
// http://en.wikipedia.org/wiki/Logistic_function
typedef enum {
    MLPRegression,     // Linear output from -Inf to +Inf
    MLPClassification, // Interval from 0 to 1
} MLPOutput;

@interface MLPNeuralNet : NSObject

// Designated initializer
- (id)initWithLayersConfig:(NSArray *)layersConfig // of NSNumbers
                   weights:(NSArray *)weights      // of NSNumbers
                outputMode:(MLPOutput)outputMode;

- (NSNumber *)predictByFeatureVector:(NSArray *)vector; // of NSNumbers

@end
