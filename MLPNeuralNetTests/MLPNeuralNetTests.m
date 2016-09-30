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
    NSData *weightsForANDModel;
    NSArray *layersForANDModel;
    MLPNeuralNet *modelOfAND;
    
    NSData *weightsForORModel;
    NSArray *layersForORModel;
    MLPNeuralNet *modelOfOR;
    
    NSData *weightsForXNORModel;
    NSArray *layersForXNORModel;
    MLPNeuralNet *modelOfXNOR;
    
    NSData *weightsForRModel;
    NSArray *layersForRModel;
    MLPNeuralNet *modelFromR;
    
    NSData *wtsForMulticlassModel;
    NSArray *layersForMulticlassModel;
    MLPNeuralNet *modelMultclass;
    
    NSData *wtsNeurlab;
    NSArray *layersForNeurolab;
    MLPNeuralNet *modelNeurolab;
    
    NSData *wtsForReLUModel;
    NSArray *layersForReLUModel;
    MLPNeuralNet *modelWithReLU;
    
    NSData *wtsForReLUSigModel;
    NSArray *layersForReLUSigModel;
    MLPNeuralNet *modelWithReLUSig;

    NSData *wtsForReLUSoftmaxModel;
    NSArray *layersForReLUSoftmaxModel;
    MLPNeuralNet *modelWithReLUSoftmax;
    
    NSData *wtsForModelWithBatchNorm;
    NSArray *layersForModelWithBatchNorm;
    NSMutableArray *layerTypesForModelWithBatchNorm;
    MLPNeuralNet *modelForModelWithBatchNorm;
    
    NSData *vector;
    NSMutableData *prediction;
    double *assessment;

    NSData *twoBinaryFeatures;
    
    NSMutableData *predictionM3;
    double *assessmentM3;
    
    NSMutableData *predictionM4;
    double *assessmentM4;
}

@end

@implementation MLPNeuralNetTests

- (void)setUp {
    [super setUp];
    double wtsForANDModel[] = {-30.0, 20.0, 20.0};
    weightsForANDModel = [NSData dataWithBytes:wtsForANDModel length:sizeof(wtsForANDModel)];
    layersForANDModel = [NSArray arrayWithObjects:@2, @1, nil];
    modelOfAND = [[MLPNeuralNet alloc] initWithLayerConfig:layersForANDModel weights:weightsForANDModel outputMode:MLPClassification];
    
    double wtsForORModel[] = {-10.0, 20.0, 20.0,};
    weightsForORModel = [NSData dataWithBytes:wtsForORModel length:sizeof(wtsForORModel)];
    layersForORModel = [NSArray arrayWithObjects:@2, @1, nil];
    modelOfOR = [[MLPNeuralNet alloc] initWithLayerConfig:layersForORModel weights:weightsForORModel outputMode:MLPClassification];
    
    double wtsForXNORModel[] = {-30, 20, 20, 10, -20, -20, -10, 20, 20};
    weightsForXNORModel = [NSData dataWithBytes:wtsForXNORModel length:sizeof(wtsForXNORModel)];
    layersForXNORModel = [NSArray arrayWithObjects:@2, @2, @1, nil];
    modelOfXNOR = [[MLPNeuralNet alloc] initWithLayerConfig:layersForXNORModel weights:weightsForXNORModel outputMode:MLPClassification];
    
    double wtsForRModel[] = {-2.71060097220996, -6.81214916465476, -12.1053278516406, 18.2715947013538, 8.72334781502212, 2.35809610106244, 15.719139731507, -1.80639409232081, 2.51353212123316, -2.4189109388074, -17.4296209203538, -10.3449714315882, -22.1011893034548, -49.9606972196645, 13.9161325812463, 6.53648231249005, -9.29399062725314, -1.85059372182408, 0.0598502514968646, 6.81377862712891, -3.68021479940558, -13.076783481441, 8.8414384652821, 1.46478402341764, -11.7921420100838, -6.16682430547976, 21.5509119968678, -7.2621449606834, -5.15464259334699, -1.0980818195238, -0.317230719329745, 16.2737913653954, -14.978757957712, -1.16743202142148, -18.1595830476094, 12.2396321827846, 7.76513511142224, -2.37688971183376, 22.8116347135894, 12.3208793627677, 26.1556354804688, -3.14149286925168, -5.58723668424285, -4.68286637029157, 7.32089244900786, -7.55981449652514, 14.5456523331983, 7.49524484679645, -5.29892473113883, -25.3634340327607, 21.443143125663, 0.216461110629665, -18.2522595249887, -28.0508966195186, -7.99909662642079, -35.7891190840656, 3.60857227606505, 33.0307886636866, 19.0713862250353, -1.58618873665141, -45.1853374078929, -30.1465630071669, 16.7637781094333, 9.67916304086313, 15.1735609326649, -15.2174905606174, 7.82173017526338, 7.68308877276538, 11.5871878739183, -15.9650199466877, 16.6350333484937, -14.8885719190494, 21.9284878578986, 9.58508061399546, 28.7820941145138, 47.3556138680817, -15.5605644509725, -30.6571481419602, -33.9379691928721, -10.1255590224846, -23.7681637619784};
    weightsForRModel = [NSData dataWithBytes:wtsForRModel length:sizeof(wtsForRModel)];
    layersForRModel = [NSArray arrayWithObjects:@6, @10, @1, nil];
    modelFromR = [[MLPNeuralNet alloc] initWithLayerConfig:layersForRModel weights:weightsForRModel outputMode:MLPClassification];
    
    
    double wtsMulticlass[] = {3.38869422183494, 2.21275111602689, 0.0364499956405933, -2.83979725768616, -2.84707419013891, -15.9947187643494, 3.95041208835756, 8.31271557337315, -4.70371970234104, -14.568533401802, -8.18033413139586, 35.0666149280761, -34.8853823433816, -5.6715805364208, -0.689952966866256, 11.4263281305969, 7.51887270965955, -30.4760492359728, -20.8005731948335};
    wtsForMulticlassModel = [NSData dataWithBytes:wtsMulticlass length:sizeof(wtsMulticlass)];
    layersForMulticlassModel = [NSArray arrayWithObjects:@4, @2, @3, nil];
    modelMultclass = [[MLPNeuralNet alloc] initWithLayerConfig:layersForMulticlassModel
                                                     weights:wtsForMulticlassModel
                                                  outputMode:MLPClassification];
    
    double wtsPythonNeuorolab[] = {-2.522616844907733, 1.379419132631195, 2.384441621038408, -4.411649980191586, -0.685608059818619, -10.887683996948859, -3.463033475731954, -3.561827798081323, 6.694420878577903, 5.847634136214969, -3.968908727857065, 8.456453936484778, -22.223450906514095, 8.545523821607908, -14.004325499443924, -16.865896703590984, -10.256283161547678, 3.198517762647665, 20.095491610370658};
    wtsNeurlab = [NSData dataWithBytes:wtsPythonNeuorolab length:sizeof(wtsPythonNeuorolab)];
    layersForNeurolab = [NSArray arrayWithObjects:@4, @2, @3, nil];
    modelNeurolab = [[MLPNeuralNet alloc] initWithLayerConfig:layersForNeurolab
                                                      weights:wtsNeurlab
                                                   outputMode:MLPClassification];
    
    double wtsForReLU[] = {-1.0, 1.0, 20.0};
    wtsForReLUModel = [NSData dataWithBytes:wtsForReLU length:sizeof(wtsForReLU)];
    layersForReLUModel = [NSArray arrayWithObjects:@2, @1, nil];
    modelWithReLU = [[MLPNeuralNet alloc] initWithLayerConfig:layersForReLUModel
                                                           weights:wtsForReLUModel
                                                        outputMode:MLPClassification];
    modelWithReLU.activationFunction = MLPReLU;
    
    double wtsForReLUSig[] = {-1.0, 1.0, 20.0};
    wtsForReLUSigModel = [NSData dataWithBytes:wtsForReLUSig length:sizeof(wtsForReLUSig)];
    layersForReLUSigModel = [NSArray arrayWithObjects:@2, @1, nil];
    modelWithReLUSig = [[MLPNeuralNet alloc] initWithLayerConfig:layersForReLUSigModel
                                                      weights:wtsForReLUSigModel
                                                   outputMode:MLPClassification];
    
    modelWithReLUSig.hiddenActivationFunction = MLPReLU;
    modelWithReLUSig.outputActivationFunction = MLPSigmoid;
    
    double wtsForReLUSoftmax[] = {-0.00333056, -0.29518637,  0.26010591,  0.00627716, -0.63008577,
                                   0.5226832 ,  0.02341191,  0.89141166, -0.66637737, -0.02341191,
                                  -0.19588685, -0.01236533};
    wtsForReLUSoftmaxModel = [NSData dataWithBytes:wtsForReLUSoftmax length:sizeof(wtsForReLUSoftmax)];
    layersForReLUSoftmaxModel = [NSArray arrayWithObjects:@2, @2, @2, nil];
    modelWithReLUSoftmax = [[MLPNeuralNet alloc] initWithLayerConfig:layersForReLUSoftmaxModel
                                                             weights:wtsForReLUSoftmaxModel
                                                          outputMode:MLPClassification];
    modelWithReLUSoftmax.hiddenActivationFunction = MLPReLU;
    modelWithReLUSoftmax.outputActivationFunction = MLPSoftmax;

    double wtsForModelWithBN[] = {-0.02911046,  0.96149528, -0.30875102, -0.07312316, -0.53453773,
                                  -0.95400345, -0.04936157, -0.60033119,  1.01297891,  0.8972764 ,
                                   0.91061068,  0.95709789,  0.0074056 ,  0.00735936, -0.00107296,
                                   0.38664621,  0.41322696,  0.35796061,  0.53616369,  0.51914299,
                                   0.56763041,  0.00775698,  0.4242098 ,  0.59139675, -0.1122655 ,
                                  -0.00775698, -0.75524676, -0.39608201, -0.15460265};
    
    wtsForModelWithBatchNorm = [NSData dataWithBytes:wtsForModelWithBN length:sizeof(wtsForModelWithBN)];
    // 2 = input shape; 3 = #neurons in dense layer; 3 = BatchNorm shape; 2 = output shape
    layersForModelWithBatchNorm = [NSArray arrayWithObjects:@2, @3, @3, @2, nil];
    layerTypesForModelWithBatchNorm = [NSMutableArray arrayWithObjects:@(MLPLayerDense), @(MLPLayerBatchNormalization), @(MLPLayerDense), nil];
    modelForModelWithBatchNorm = [[MLPNeuralNet alloc] initWithLayerConfigAndLayerType:layersForModelWithBatchNorm
                                                                               weights:wtsForModelWithBatchNorm
                                                                            layerTypes:layerTypesForModelWithBatchNorm
                                                                            outputMode:MLPClassification];
    modelForModelWithBatchNorm.hiddenActivationFunction = MLPReLU;
    modelForModelWithBatchNorm.outputActivationFunction = MLPSoftmax;

    
    double features[] = {
        1, 1,
        1, 0,
        0, 1,
        0, 0};
    twoBinaryFeatures = [NSData dataWithBytes:features length:sizeof(features)];
    
    prediction = [NSMutableData dataWithLength:sizeof(double)];
    assessment = (double *)prediction.bytes;
    
    predictionM3 = [NSMutableData dataWithLength:sizeof(double)*3];
    assessmentM3 = (double *)predictionM3.bytes;
    
    predictionM4 = [NSMutableData dataWithLength:sizeof(double)*4];
    assessmentM4 = (double *)predictionM4.bytes;
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

#pragma mark - AND model tests

- (void)testModelOfANDOneOne {
    double features[] = {1, 1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfAND predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfANDOneZero {
    double features[] = {1, 0};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfAND predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelOfANDZeroOne {
    double features[] = {0, 1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfAND predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelOfANDZeroZero {
    double features[] = {0, 0};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfAND predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

#pragma mark - OR model tests

- (void)testModelOfOROneOne {
    double features[] = {1, 1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfOROneZero {
    double features[] = {1, 0};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfORZeroOne {
    double features[] = {0, 1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfORZeroZero {
    double features[] = {0, 0};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

#pragma mark - XNOR model tests

- (void)testModelOfXNOROneOne {
    double features[] = {1, 1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

- (void)testModelOfXNOROneZero {
    double features[] = {1, 0};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelOfXNORZeroOne {
    double features[] = {0, 1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelOfXNORZeroZero {
    double features[] = {0, 0};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1, 0.0001);
}

#pragma mark - AND matrix model tests

- (void)testModelOfANDMatrix {
    [modelOfAND predictByFeatureMatrix:twoBinaryFeatures intoPredictionMatrix:predictionM4];
    XCTAssertEqualWithAccuracy(assessmentM4[0], 1, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[1], 0, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[2], 0, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[3], 0, 0.0001);
}

#pragma mark - OR matrix model tests

- (void)testModelOfORMatrix {
    [modelOfOR predictByFeatureMatrix:twoBinaryFeatures intoPredictionMatrix:predictionM4];
    XCTAssertEqualWithAccuracy(assessmentM4[0], 1, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[1], 1, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[2], 1, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[3], 0, 0.0001);
}


#pragma mark - XNOR matrix model tests

- (void)testModelOfXNORMatrix {
    [modelOfXNOR predictByFeatureMatrix:twoBinaryFeatures intoPredictionMatrix:predictionM4];
    XCTAssertEqualWithAccuracy(assessmentM4[0], 1, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[1], 0, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[2], 0, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM4[3], 1, 0.0001);
}

#pragma mark - R model portability

- (void)testPortabilityofModelFromR {
    double features[] = {-0.1469695, 0.3815642, 0.9089234, 0.0756491, 0.03446598, 0.005667798};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelFromR predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0.9999989, 0.0000001);
}

- (void)testMulticlassModelFromR {
    double features[] = {4.8, 3.3, 1.3, 0.2};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelMultclass predictByFeatureVector:vector intoPredictionVector:predictionM3];
    XCTAssertEqualWithAccuracy(assessmentM3[0], 0.0003350431, 0.0000001);
    XCTAssertEqualWithAccuracy(assessmentM3[1], 0.9937246, 0.0000001);
    XCTAssertEqualWithAccuracy(assessmentM3[2], 0, 0.0000001);
}

#pragma mark - Python Neurolab compatability

- (void)testPythonNeurolabModel {
    double features[] = {4.8,  3.3,  1.3,  0.2};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelNeurolab predictByFeatureVector:vector intoPredictionVector:predictionM3];
    XCTAssertEqualWithAccuracy(assessmentM3[0], 9.88665737e-01, 0.000000001);
    XCTAssertEqualWithAccuracy(assessmentM3[1], 4.37569362e-03, 0.000000001);
    XCTAssertEqualWithAccuracy(assessmentM3[2], 8.53800274e-04, 0.000000001);
}


#pragma mark - Activation functions

- (void)testModelWithTangentOutput {
    double features[] = {4.8,  3.3,  1.3,  0.2};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    modelNeurolab.activationFunction = MLPTangent;
    [modelNeurolab predictByFeatureVector:vector intoPredictionVector:predictionM3];
    XCTAssertEqualWithAccuracy(assessmentM3[0], 1, 0.000000001);
    XCTAssertEqualWithAccuracy(assessmentM3[1], 0.999999999753, 0.000000001);
    XCTAssertEqualWithAccuracy(assessmentM3[2], -1, 0.000000001);
}

- (void)testModelWithReLUBelowThreshold {
    double features[] = {-1, -1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelWithReLU predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 0, 0.0001);
}

- (void)testModelWithReLUAboveThreshold {
    double features[] = {-1, 1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelWithReLU predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 18.0, 0.0001);
}

- (void)testModelWithDifferentHiddenVsOutputActivation {
    double features[] = {-1, 1};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    [modelWithReLUSig predictByFeatureVector:vector intoPredictionVector:prediction];
    XCTAssertEqualWithAccuracy(assessment[0], 1.0, 0.0001);
}

- (void)testModelSettingsWithDifferentActivations {
    XCTAssertEqual(modelWithReLUSig.activationFunction, MLPReLU);
    XCTAssertEqual(modelWithReLUSig.hiddenActivationFunction, MLPReLU);
    XCTAssertEqual(modelWithReLUSig.outputActivationFunction, MLPSigmoid);
}

#pragma mark - Number of weigths

- (void)testNumberOfWeigthsByLayerConfig {
    NSArray *cfg = @[@2, @3, @2, @1];
    NSMutableArray *layer_types = [NSMutableArray arrayWithObjects:@(MLPLayerDense), @(MLPLayerDense), @(MLPLayerDense), nil];
    XCTAssertEqual([MLPNeuralNet countWeights:cfg
                                  layersTypes:layer_types], (NSInteger)20);
    
    NSArray *cfg2 = @[@2, @3, @2, @1];
    NSMutableArray *layer_types2 = [NSMutableArray arrayWithObjects:@(MLPLayerDense), @(MLPLayerBatchNormalization), @(MLPLayerDense), nil];
    XCTAssertEqual([MLPNeuralNet countWeights:cfg2
                                  layersTypes:layer_types2], (NSInteger)24);
}

#pragma mark - Exception tests

- (void)testWrongFeatureVectorSize {
    double features[] = {0, 0};
    vector = [NSData dataWithBytes:features length:sizeof(features)-1];
    XCTAssertThrowsSpecificNamed([modelOfXNOR predictByFeatureVector:vector intoPredictionVector:prediction],
                                 NSException,
                                 @"NSInternalInconsistencyException");
}

- (void)testWrongPredictionVectorSize {
    double features[] = {0, 0};
    vector = [NSData dataWithBytes:features length:sizeof(features)-1];
    NSMutableData *predictionVector = [NSMutableData dataWithLength:sizeof(double)-1];
    XCTAssertThrowsSpecificNamed([modelOfXNOR predictByFeatureVector:vector intoPredictionVector:predictionVector],
                                 NSException,
                                 @"NSInternalInconsistencyException");
}

- (void)testIncorrectNumberOfWeights {
    double wts[] = {3.0, 1.0, 2.0};
    NSArray *layerCfg = @[@2, @3, @1];
    NSData *weights = [NSData dataWithBytes:wts length:sizeof(wts)];
    XCTAssertThrowsSpecificNamed([[MLPNeuralNet alloc] initWithLayerConfig:layerCfg weights:weights outputMode:MLPRegression],
                                 NSException,
                                 @"NSInternalInconsistencyException");
}

- (void)testSoftmaxOutputLayer {
    double features[] = {-1, 10};
    
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    NSMutableData* predictionM2 = [NSMutableData dataWithLength:sizeof(double) * 2];
    double* assessmentM2 = (double *)predictionM2.bytes;
    
    [modelWithReLUSoftmax predictByFeatureMatrix:vector intoPredictionMatrix:predictionM2];
    XCTAssertEqualWithAccuracy(assessmentM2[0], 0.3447236, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM2[1], 0.6552764, 0.0001);
}

-(void)testBatchNormalizationLayer {
    double features[] = {-1, 10};
    vector = [NSData dataWithBytes:features length:sizeof(features)];
    
    NSMutableData* predictionM2 = [NSMutableData dataWithLength:sizeof(double) * 2];
    double* assessmentM2 = (double *)predictionM2.bytes;
    
    [modelForModelWithBatchNorm predictByFeatureMatrix:vector intoPredictionMatrix:predictionM2];
    XCTAssertEqualWithAccuracy(assessmentM2[0], 0.32947651, 0.0001);
    XCTAssertEqualWithAccuracy(assessmentM2[1], 0.67052352, 0.0001);
}

@end
