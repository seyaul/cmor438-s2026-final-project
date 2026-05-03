import numpy as np

#do we need pandas currently?
# import pandas as pd

#below 4 should belong to our jupyter notebook? 
# from sklearn.datasets import load_iris
# # from rice_ML.supervised_ml.Perceptron import perceptron
# from rice_ML.preprocessing import train_test_split, standardize
# from rice_ML.preprocessing import accuracy_score

import matplotlib.pyplot as plt


class Perceptron(object):
    #eta is learning rate / our neuron cost function derivative 1/2 * 2 
    def __init__(self, eta = 0.1, epochs = 50):
        self.eta = eta
        self.epochs = epochs

#write a high level program for what needs to be done 

    def train(self, X, y):

        #initialize w1, w2, and b
        #np.random.rand(1+X.shape[1]) = septal width randomized
        self.w_b = np.random.rand(1 + X.shape[1])
        
        #wb will be our weights
        epoch_counter = 0

        self.mistakes = []


        #this gets the 1 x 2 matrix x and y and assigns it into [[l1,w1],species]
        while epoch_counter < self.epochs:
            errors = 0
            for xi, yi in zip(X,y):
            
            #also mistakes

                #our alpha adjustment is the value we multiply by 
                # bias = alpha * xi
                # weight = alpha 
                prediction = self.predict(xi)
                if prediction - yi != 0:
                    adjustment = self.eta*(prediction-yi)
                    self.w_b[:-1] -= adjustment*xi
                    self.w_b[-1] -= adjustment
                    #no idea what this does
                    errors += int(adjustment != 0)
            #no idea? 
            if errors == 0:
                    return self
            else:
                self.mistakes.append(errors)
            epoch_counter += 1
            
        return 

    #product length and width by the weights of each + bias
    def net_input(self, X):
        return np.dot(X, self.w_b[:-1]) + self.w_b[-1]

    # if our net input is > 0, then 1. yeah. based on line 26.
    def predict(self, X):
        return np.where(self.net_input(X) >=0, 1, -1)