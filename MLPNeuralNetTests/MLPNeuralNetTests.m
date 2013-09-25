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
    NSArray *ANDModelWeigths;
    NSArray *ANDModelLayers;
    MLPNeuralNet *ANDmodel;
}

@end

@implementation MLPNeuralNetTests

- (void)setUp
{
    [super setUp];
    ANDModelWeigths = [NSArray arrayWithObjects:@-30, @20, @20, nil];
    ANDModelLayers = [NSArray arrayWithObjects:@2, @1, nil];
    ANDmodel = [[MLPNeuralNet alloc] initWithLayersConfig:ANDModelLayers weights:ANDModelWeigths outputMode:MLPClassification];
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

- (void)testANDModelOneOne
{
    double assessment = [[ANDmodel predictByFeatureVector:@[@1, @1]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 1, 0.0001);
}

- (void)testANDModelOneZero
{
    double assessment = [[ANDmodel predictByFeatureVector:@[@1, @0]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

- (void)testANDModelZeroOne
{
    double assessment = [[ANDmodel predictByFeatureVector:@[@0, @1]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

- (void)testANDModelZeroZero
{
    double assessment = [[ANDmodel predictByFeatureVector:@[@0, @0]] doubleValue];
    XCTAssertEqualWithAccuracy(assessment, 0, 0.0001);
}

@end
