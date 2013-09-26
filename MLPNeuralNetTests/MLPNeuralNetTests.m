//
//  MLPNeuralNetTests.m
//  MLPNeuralNetTests
//
//  Created by Mykola Pavlov on 9/25/13.
//  Copyright (c) 2013 Biomech, Inc. All rights reserved.
//

#import <XCTest/XCTest.h>
#import "MLPNeuralNet.h"

@interface MLPNeuralNetTests : XCTestCase {
    NSArray *weightsForANDModel;
    NSArray *layersForANDModel;
    MLPNeuralNet *modelOfAND;
    
    NSArray *weightsForORModel;
    NSArray *layersForORModel;
    MLPNeuralNet *modelOfOR;
    
    NSArray *weightsForXNORModel;
    NSArray *layersForXNORModel;
    MLPNeuralNet *modelOfXNOR;
    
    NSMutableData *vector;
    NSMutableData *prediction;
    double *assessment;
}

@end

@implementation MLPNeuralNetTests

- (void)setUp
{
    [super setUp];
    weightsForANDModel = [NSArray arrayWithObjects:@-30, @20, @20, nil];
    layersForANDModel = [NSArray arrayWithObjects:@2, @1, nil];
    modelOfAND = [[MLPNeuralNet alloc] initWithLayersConfig:layersForANDModel weights:weightsForANDModel outputMode:MLPClassification];
    
    weightsForORModel = [NSArray arrayWithObjects:@-10, @20, @20, nil];
    layersForORModel = [NSArray arrayWithObjects:@2, @1, nil];
    modelOfOR = [[MLPNeuralNet alloc] initWithLayersConfig:layersForORModel weights:weightsForORModel outputMode:MLPClassification];
    
    weightsForXNORModel = [NSArray arrayWithObjects:@-30, @20, @20, @10, @-20, @-20, @-10, @20, @20, nil];
    layersForXNORModel = [NSArray arrayWithObjects:@2, @2, @1, nil];
    modelOfXNOR = [[MLPNeuralNet alloc] initWithLayersConfig:layersForXNORModel weights:weightsForXNORModel outputMode:MLPClassification];
    
    prediction = [NSMutableData dataWithLength:sizeof(double)];
    assessment = (double *)prediction.bytes;
}

- (void)tearDown
{
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

#pragma mark - AND model tests

- (void)testModelOfANDOneOne
{
    double features[] = {1, 1};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfAND predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfANDOneZero
{
    double features[] = {1, 0};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfAND predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelOfANDZeroOne
{
    double features[] = {0, 1};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfAND predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelOfANDZeroZero
{
    double features[] = {0, 0};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfAND predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

#pragma mark - OR model tests

- (void)testModelOfOROneOne
{
    double features[] = {1, 1};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfOROneZero
{
    double features[] = {1, 0};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfORZeroOne
{
    double features[] = {0, 1};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfORZeroZero
{
    double features[] = {0, 0};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

#pragma mark - XNOR model tests

- (void)testModelOfXNOROneOne
{
    double features[] = {1, 1};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfXNOROneZero
{
    double features[] = {1, 0};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelOfXNORZeroOne
{
    double features[] = {0, 1};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelOfXNORZeroZero
{
    double features[] = {0, 0};
    vector = [NSMutableData dataWithBytes:features length:sizeof(features)];
    [modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

// TODO
// Test that weight matrix conforms to network configuration

@end
