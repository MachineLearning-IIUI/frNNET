# -*- coding: utf-8 -*-
"""
The goal of this file is to compare fractional and normal Neural Networks

@author: Sharjeel Abid Butt

@References

1. 
"""

import mnist_load as mload
import copy

import numpy as np
#import scipy as sp
#import scipy.stats as stats
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import gamma

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

def dataExtraction(data = 'train', class1 = 1, class0 = 0):
    
    [data, labels] = mload.load(data)
    y1 = np.extract(labels == class1, labels)
    X1 = data[labels == class1, :, :]
    
    y0 = np.extract(labels == class0, labels)
    X0 = data[labels == class0, :, :]

    y = np.concatenate((y1, y0), axis = 0)
    X = np.concatenate((X1, X0), axis = 0)
    
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1] * np.shape(X)[2]))
    X = (X - np.mean(X, axis = 0)) / (1 + np.std(X, axis = 0)) # Data Normalization
    y[y == class1] = 1
    y[y == class0] = 0
    y = np.reshape(y, (np.shape(X)[0], 1))
    return y, X



class nnet(object):
    """
    A class that implements Basic Neural Networks Architecture
    """
    
    def __init__(self, noOfInputs = 2, noOfLayers = 2, nodesInEachLayer = [2, 2], 
               noOfOutputs = 2, activationFunction = 'sigmoid', fr = 1, 
               parametersRange = [-1, 1]):
        """
        Creates a Neural Network
        """
        if (len(nodesInEachLayer) != noOfLayers):
            raise ValueError('Incorrect Parameters provided!')
        
        self.n_I       = noOfInputs
        self.n_L       = noOfLayers
        self.n_H       = nodesInEachLayer
        self.n_O       = noOfOutputs
        self.a_Func    = activationFunction
        self.pR        = parametersRange
        self.fr        = fr
        
        #self.Nstruct   = [noOfInputs, nodesInEachLayer, noOfOutputs]
        self.Theta     = []
        self.nodes     = []
#        self.nodes.append(np.zeros((noOfInputs, 1)))
        lmin, lmax     = parametersRange
        
        
        for l in range(noOfLayers + 1):
            if l == 0:
                tempTheta = self.randTheta(lmin, lmax, noOfInputs, self.n_H[l])
            elif l == noOfLayers:
                tempTheta = self.randTheta(lmin, lmax, self.n_H[-1], noOfOutputs)
            else:
                tempTheta = self.randTheta(lmin, lmax, self.n_H[l - 1], self.n_H[l])
            
            tempNode  = np.shape(tempTheta)[1]
            
            self.Theta.append(tempTheta)
            self.nodes.append(tempNode)
    
    def __str__(self):
        return "This neural network has a " + str(self.n_I) + ' X ' + \
        str(self.n_H) + ' X ' + str(self.n_O) + " structure."
        
    def randTheta(self, l_min, l_max, i_nodes, o_nodes):
        theta = l_min + np.random.rand(i_nodes + 1, o_nodes) * (l_max - l_min)
        return theta
    
    def sigmoid(self, z, derivative = False):
        if derivative:
            return z * (1 - z)
        S = 1.0 / (1.0 + np.exp(-z))
        return S
        
    def setTheta(self, thetaLayer = 1, thetaIndex = [1, 0], allSet = False):
        """
        Updates the Theta values of the Neural Network
        """
        if allSet:
            for l in range(self.n_L + 1):
                print '\n\nEnter Theta values for Layer ' + str(l + 1) + ':\n'
                in_nodes, out_nodes = np.shape(self.Theta[l])
                for inIndex in range(in_nodes):
                    for outIndex in range(out_nodes):
                        self.Theta[l][inIndex][outIndex] = float (raw_input( \
                        'Enter Theta[' + str(outIndex + 1) + '][' + str(inIndex) +']:'))
        else:
            outIndex, inIndex = thetaIndex
            self.Theta[thetaLayer - 1][inIndex][outIndex - 1] = float (raw_input( \
                        'Enter Theta[' + str(outIndex) + '][' + str(inIndex) +']:'))
        
        print '\n\n\nTheta Update Complete.\n\n\n'
        
    def getTheta(self):
        return copy.deepcopy(self.Theta)
    
    def forward_pass(self, nodes, X, y):
        """
        Does the forward pass stage of Backpropagation
        """
#        raise NotImplementedError
        m = np.shape(y)[0]
        
        for l in range(self.n_L + 1):
            if l == 0:
                node_in  = np.concatenate((np.ones((m, 1)), X), axis = 1)
            else:
                node_in  = np.concatenate((np.ones((m, 1)), nodes[l - 1]), axis = 1)
            node_out = np.dot(node_in, self.Theta[l])
                
            if self.a_Func == 'sigmoid':
                nodes[l] = self.sigmoid(node_out)
            
        return nodes
                        
    
    def backward_pass(self, delta, nodes, X, y, grad, Lambda, quadLoss):
        """        
        Does the Backpass stage of Backpropagation        
        """
#        raise NotImplementedError    
        m = np.shape(y)[0]
        
        if self.a_Func == 'sigmoid':
            delta[-1] = (nodes[-1] - y)
            if quadLoss:
                delta[-1] *= self.sigmoid(nodes[-1], True)
            for l in range(self.n_L - 1, -1, -1):
                delta[l] = np.dot(delta[l + 1], self.Theta[l + 1][1:].T) \
                           * self.sigmoid(nodes[l], True)

        for l in range(self.n_L + 1):
            if l == 0:
                Xconcate = np.concatenate((np.ones((m, 1)), X), axis = 1)
                grad[l]  = np.dot(Xconcate.T, delta[l]) 
            else:
                nodeConcated = np.concatenate((np.ones((m, 1)), nodes[l - 1]), axis = 1)
                grad[l]      = np.dot(nodeConcated.T, delta[l]) 
            
            if Lambda != 0:
                    grad[l][1:] += Lambda * self.Theta[l][1:]
            
        return grad, delta
                
    
    def trainNNET(self, data, labels, stoppingCriteria = 1e-3, LearningRate = 1e-1, 
                  Lambda = 0, noOfIterations = 1000, quadLoss = False, moreDetail = False):
        """
        Does the training of the Neural Network
        """
#        raise NotImplementedError
        if (np.shape(data)[0]   != np.shape(labels)[0] or \
            np.shape(data)[1]   != self.n_I or \
            np.shape(labels)[1] != self.n_O):
                raise ValueError('Data is not suitable for this neural network')
        
        m     = np.shape(labels)[0]
        nodes = []
        delta = []
        grad  = []
        eV    = []
        
        print 'Training Started:'
        for l in range(self.n_L + 1):            
            nodes.append(np.zeros((m, self.nodes[l])))
            delta.append(np.zeros((m, self.nodes[l])))
            grad.append(np.shape(self.Theta[l]))

        print "Epoch \t Error"
        for epoch in range(noOfIterations):
            nodes = self.forward_pass(nodes, data, labels)
            
            labels_hat = nodes[-1]
            
            if quadLoss:
                error = np.sum((labels_hat - labels) ** 2) / (2.0 * m)
            else:
                error = - np.sum(labels * np.log(labels_hat) + \
                       (1.0 - labels) * np.log(1.0 - labels_hat)) * 1.0 / m           
            if Lambda != 0:
                for l in range(self.n_L + 1):
                    error += Lambda / 2 * np.sum(self.Theta[l][1:] ** 2) / m
                
            print str(epoch) + " \t " + str(np.nan_to_num(error))
            
            eV.append(error)
            
            if error <= stoppingCriteria:
                break
            else:
                grad, delta = self.backward_pass(delta, nodes, data, \
                                                 labels, grad, Lambda, quadLoss)
                
                for l in range(self.n_L + 1):
                    self.Theta[l] -= LearningRate / (m * gamma(2 - self.fr)) * \
                              grad[l] * np.power(np.abs(self.Theta[l]), (1 - self.fr))                       
                    
        if moreDetail:
            return eV, nodes, grad, delta
            
        return eV
        

    def predictNNET(self, data, labels):
        nodes = []        
        m     = np.shape(labels)[0]
        for l in range(self.n_L + 1):            
            nodes.append(np.zeros((m, self.nodes[l])))
        
        nodes = self.forward_pass(nodes, data, labels)
        
        labels_hat = nodes[-1] > 0.5
        return labels_hat    
     




# Main Code starts here

class1 = 1
class0 = 0

labelsTrain, dataTrain = dataExtraction('train', class1, class0)

noOfIter     = 100
#learningRate = 1e-1
stopVal      = 1e-3
Lambda       = 1e-1
pR           = [-1, 1]

LRVec       = [0.4]
nnColumns   = ["NNET", "eV", "TrainAccuracy", "TestAccuracy"]
nnClassifier = pd.DataFrame(data = np.zeros((len(LRVec), len(nnColumns))), dtype = list, \
                            columns = nnColumns)


for ind in range(len(LRVec)):
    np.random.seed(0)
    
    nnClassifier.NNET[ind] = nnet(noOfInputs = 784, noOfLayers = 2, \
                     nodesInEachLayer = [50, 50], noOfOutputs = 1, \
                     parametersRange = pR)
    tic()
    nnClassifier.eV[ind] = nnClassifier.NNET[ind].trainNNET(dataTrain, labelsTrain, 
                           noOfIterations = noOfIter, LearningRate = LRVec[ind], \
                           Lambda = Lambda, quadLoss = False)
    toc()

    #plt.figure()
    #plt.plot(loss)
    #plt.show()
    
    print "\n\n\n"
    
    labels_hatTrain = nnClassifier.NNET[ind].predictNNET(dataTrain, labelsTrain)
    nnClassifier.TrainAccuracy[ind]  = np.sum(labels_hatTrain == labelsTrain) * 100.0 / np.shape(labelsTrain)[0]
    print "Training Accuracy = " + str(nnClassifier.TrainAccuracy[ind]) + "%"
    
    labelsTest, dataTest = dataExtraction('test', class1, class0)
    
    labels_hatTest = nnClassifier.NNET[ind].predictNNET(dataTest, labelsTest)
    nnClassifier.TestAccuracy[ind]  = np.sum(labels_hatTest == labelsTest) * 100.0 / np.shape(labelsTest)[0]
    print "Test Accuracy = " + str(nnClassifier.TestAccuracy[ind]) + "%"

                              
# Fractional Neural Networks
fr        = [1.3, 1.5]
frColumns = ["NNET", "eV", "TrainAccuracy", "TestAccuracy"]

frClassifier = list()
for ind in range(len(LRVec)):
        frClassifier.append(pd.DataFrame(data = np.zeros((len(fr), len(frColumns))),\
                    dtype = list, columns = frColumns))
#frClassifier = pd.DataFrame({"NNET" : np.ndarray(len(fr)),
#                             "eV" : np.float64(len(fr)), 
#                             "TrainAccuracy" : np.float64(len(fr)),
#                              "TestAccuracy" : np.float64(len(fr))})

for ind1 in range(len(LRVec)):
    for ind2 in range(len(fr)):
        np.random.seed(0)
        
        frClassifier[ind1].NNET[ind2] = nnet(noOfInputs = 784, noOfLayers = 2, \
                    nodesInEachLayer = [50, 50], noOfOutputs = 1, parametersRange = pR, \
                    fr = fr[ind2])
        
        tic()
        frClassifier[ind1].eV[ind2] = frClassifier[ind1].NNET[ind2].trainNNET(dataTrain, \
                    labelsTrain, noOfIterations = noOfIter, LearningRate = LRVec[ind1], \
                    Lambda = Lambda, quadLoss = False)
        toc()
    #    plt.figure()
    #    plt.plot(frClassifier.eV[ind])
    #    plt.show()
    #    
        print "\n\n\n"
        
        labels_hatTrain                         = frClassifier[ind1].NNET[ind2].predictNNET(dataTrain, labelsTrain)
        frClassifier[ind1].TrainAccuracy[ind2]  = np.sum(labels_hatTrain == labelsTrain) * 100.0 \
                                  / np.shape(labelsTrain)[0]
        print "Training Accuracy = " + str(frClassifier[ind1].TrainAccuracy[ind2]) + "%"
    
            
        labels_hatTest                  = frClassifier[ind1].NNET[ind2].predictNNET(dataTest, labelsTest)
        frClassifier[ind1].TestAccuracy[ind2]  = np.sum(labels_hatTest == labelsTest) * 100.0 \
                                  / np.shape(labelsTest)[0]
        print "Test Accuracy = " + str(frClassifier[ind1].TestAccuracy[ind2]) + "%"

plt.close('all')    
for ind1 in range(len(LRVec)):
    plt.figure()
    plt.hold(True)
    
    plt.plot(nnClassifier.eV[ind], label = 'fr = 1')
    
    
    for ind2 in range(len(fr)):
        plt.plot(frClassifier[ind1].eV[ind2], label = 'fr = ' +  str(fr[ind2]))
    
    plt.hold(False)
    plt.title('Learning rate = ' + str(LRVec[ind1]))
    plt.grid('on')
    plt.legend()
    plt.show()

