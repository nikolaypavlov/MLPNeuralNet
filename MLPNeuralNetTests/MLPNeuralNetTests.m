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
}

- (void)tearDown
{
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

- (void)testImproperInitialization
{
    NSArray *weigths = [NSArray arrayWithObjects:[NSArray array], @1, nil];
    NSArray *layers = [NSArray arrayWithObjects:@2, @3, @1, nil];
    XCTAssertThrowsSpecificNamed([[MLPNeuralNet alloc] initWithLayersConfig:layers weights:weigths outputMode:MLPClassification],
                                 NSException,
                                 @"MLPNeuralNet initializer");
}

#pragma mark - AND model tests

- (void)testModelOfANDOneOne
{
    double assessment = [[modelOfAND predictByFeatureVector:@[@1, @1]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 1, 0.0001);
}

- (void)testModelOfANDOneZero
{
    double assessment = [[modelOfAND predictByFeatureVector:@[@1, @0]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

- (void)testModelOfANDZeroOne
{
    double assessment = [[modelOfAND predictByFeatureVector:@[@0, @1]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

- (void)testModelOfANDZeroZero
{
    double assessment = [[modelOfAND predictByFeatureVector:@[@0, @0]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

#pragma mark - OR model tests

- (void)testModelOfOROneOne
{
    double assessment = [[modelOfOR predictByFeatureVector:@[@1, @1]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 1, 0.0001);
}

- (void)testModelOfOROneZero
{
    double assessment = [[modelOfOR predictByFeatureVector:@[@1, @0]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 1, 0.0001);
}

- (void)testModelOfORZeroOne
{
    double assessment = [[modelOfOR predictByFeatureVector:@[@0, @1]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 1, 0.0001);
}

- (void)testModelOfORZeroZero
{
    double assessment = [[modelOfOR predictByFeatureVector:@[@0, @0]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

#pragma mark - XNOR model tests

- (void)testModelOfXNOROneOne
{
    double assessment = [[modelOfXNOR predictByFeatureVector:@[@1, @1]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 1, 0.0001);
}

- (void)testModelOfXNOROneZero
{
    double assessment = [[modelOfXNOR predictByFeatureVector:@[@1, @0]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

- (void)testModelOfXNORZeroOne
{
    double assessment = [[modelOfXNOR predictByFeatureVector:@[@0, @1]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

- (void)testModelOfXNORZeroZero
{
    double assessment = [[modelOfXNOR predictByFeatureVector:@[@0, @0]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 1, 0.0001);
}

// TODO
// Test that weight matrix conforms to network configuration

@end
