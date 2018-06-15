# -*- coding: utf-8 -*-
"""
NN - Number classifier

784+1 -> 100+1 -> 50+1 ->  25+1 -> 10

    100x785  50x76    25x51    10x26
"""

#%%
#Parameters
l1Size = 784
l2Size = 100
l3Size = 50
l4Size = 25
l5Size = 10

numSamples = 42000

regRate = (1/3)
learningRate = .05
iterations = 5000
#%%
import pandas as pd
import numpy as np
import time

data = pd.read_csv('train.csv', dtype='float64', nrows = numSamples)
predata = pd.read_csv('test.csv', dtype='float64')

predata=predata.values

numSamples = data.shape[0]
trainSamples = int(numSamples * 0.6)
cvSamples = int(numSamples * 0.25)

#%%
#Process Data
X = data.iloc[list(range(0,trainSamples)),list(range(1,785))].values
Y = data.iloc[list(range(0,trainSamples)), 0].values

xcv = data.iloc[list(range(trainSamples,trainSamples+cvSamples)),list(range(1,785))].values
ycv = data.iloc[list(range(trainSamples,trainSamples+cvSamples)), 0].values

#normalize data
def normData(X, mean, std):
    return (X-mean) / std

meanX = np.mean(X)
stdX = np.std(X)

X = normData(X, meanX, stdX)
xcv = normData(xcv, meanX, stdX)
predata = normData(predata, meanX, stdX)

#%%
#initialize wts

def loadThetaMatrix(fName):
    t = pd.read_csv(fName)
    return t.values

sizet1 = (l2Size,l1Size+1)
sizet2 = (l3Size,l2Size+1)
sizet3 = (l4Size,l3Size+1)
sizet4 = (l5Size,l4Size+1)

#range of wts
epst1 = np.sqrt(6) / np.sqrt(sizet1[0] + sizet1[1])
epst2 = np.sqrt(6) / np.sqrt(sizet2[0] + sizet2[1])
epst3 = np.sqrt(6) / np.sqrt(sizet3[0] + sizet3[1])
epst4 = np.sqrt(6) / np.sqrt(sizet4[0] + sizet4[1])

theta1 = np.random.random_sample(sizet1) * (2*epst1) - epst1
theta2 = np.random.random_sample(sizet2) * (2*epst2) - epst2
theta3 = np.random.random_sample(sizet3) * (2*epst3) - epst3
theta4 = np.random.random_sample(sizet4) * (2*epst4) - epst4

theta1 = loadThetaMatrix('t1')
theta2 = loadThetaMatrix('t2')
theta3 = loadThetaMatrix('t3')
theta4 = loadThetaMatrix('t4')
#%%
def sigmoid(X, grad=False):
    if grad:
        sigX = sigmoid(X)
        return np.multiply(sigX, 1-sigX)
    else:    
        return 1 / ( 1+ np.exp(-1*X))
#%%
def predict(X, theta1, theta2, theta3, theta4):
    m = X.shape[0]
 
    
    #Feed-forward
    a1 = np.append(np.ones((m, 1)), X, axis=1)
    
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((a2.shape[0], 1)), a2, axis=1)
    
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    a3 = np.append(np.ones((a3.shape[0], 1)), a3, axis=1)
    
    z4 = a3 @ theta3.T
    a4 = sigmoid(z4)
    a4 = np.append(np.ones((a4.shape[0], 1)), a4, axis=1)
    
    z5 = a4 @ theta4.T
    a5 = sigmoid(z5)
    
    
    predictions = np.argmax(a5,axis=1)
        
    return predictions
    

def costJ(X, Y, theta1, theta2, theta3, theta4, regRate):
    #helpful vars
    J = 0.0
    theta1Grads = np.zeros(theta1.shape)
    theta2Grads = np.zeros(theta2.shape)
    theta3Grads = np.zeros(theta3.shape)
    theta4Grads = np.zeros(theta4.shape)
    
    m = X.shape[0]
    
    
    #Feed-forward
    a1 = np.append(np.ones((m, 1)), X, axis=1)
    
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((a2.shape[0], 1)), a2, axis=1)
    
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    a3 = np.append(np.ones((a3.shape[0], 1)), a3, axis=1)
    
    z4 = a3 @ theta3.T
    a4 = sigmoid(z4)
    a4 = np.append(np.ones((a4.shape[0], 1)), a4, axis=1)
    
    z5 = a4 @ theta4.T
    a5 = sigmoid(z5)
    
    for i in range(0,m):
        hx = a5[i,:]
        
        num = int(Y[i])
        
        ans = np.zeros(10)
        ans[num] = 1
        
        J = J - np.sum(np.multiply(np.log(hx), ans) + np.multiply(np.log(1-hx), 1-ans))
    
    J = J / m
    
    #cost regularization (j + all nonbias thetas^2 * (regRate/2m))
    
    nonbiasT1 = theta1[:, 1:theta1.shape[1]]
    nonbiasT2 = theta2[:, 1:theta2.shape[1]]
    nonbiasT3 = theta3[:, 1:theta3.shape[1]]
    nonbiasT4 = theta4[:, 1:theta4.shape[1]]
    
    totalTheta = np.sum(np.power(nonbiasT1,2)) + np.sum(np.power(nonbiasT2,2)) 
    + np.sum(np.power(nonbiasT3,2)) + np.sum(np.power(nonbiasT4,2))
    
    reg = totalTheta * (regRate/(2*m))
    
    J = J + reg
    
    #Gradients
    for i in range(0,m):
    
        a1i = a1[[i]]
        a2i = a2[[i]]
        a3i = a3[[i]]
        a4i = a4[[i]]
        a5i = a5[[i]]
        
        
        z2i = z2[i,:].T
        z3i = z3[i,:].T
        z4i = z4[i,:].T
        
        num = int(Y[i])
        
        ans = np.zeros((10,1))
        ans[num] = 1
        
        d5 = a5i.T - ans
        
        
        i4 = theta4.T @ d5 #a4 partial deriv
        i4 = np.delete(i4, 0) #remove bias deriv
        d4 = np.multiply(i4,  sigmoid(z4i, grad=True))
        d4 = np.reshape(d4, (d4.size,1))
        
        i3 = theta3.T @ d4 #a3 partial deriv
        i3 = np.delete(i3, 0) #remove bias deriv
        d3 = np.multiply(i3,  sigmoid(z3i, grad=True))
        d3 = np.reshape(d3, (d3.size,1))
        
        i2 = theta2.T @ d3 #a2 partial deriv
        i2 = np.delete(i2, 0) #remove bias deriv
        d2 = np.multiply(i2,  sigmoid(z2i, grad=True))
        d2 = np.reshape(d2, (d2.size,1))
        
        
        theta1Grads = theta1Grads + d2 @ a1i
        theta2Grads = theta2Grads + d3 @ a2i
        theta3Grads = theta3Grads + d4 @ a3i
        theta4Grads = theta4Grads + d5 @ a4i
        
    theta1Grads = theta1Grads / m
    theta2Grads = theta2Grads / m
    theta3Grads = theta3Grads / m
    theta4Grads = theta4Grads / m
    
    #regularized gradients ( add nonbias thetas * (regRate/m) )
    
    theta1Grads[:, 1:theta1Grads.shape[1]] = theta1Grads[:, 1:theta1Grads.shape[1]] + (regRate/m) * nonbiasT1
    theta2Grads[:, 1:theta2Grads.shape[1]] = theta2Grads[:, 1:theta2Grads.shape[1]] + (regRate/m) * nonbiasT2
    theta3Grads[:, 1:theta3Grads.shape[1]] = theta3Grads[:, 1:theta3Grads.shape[1]] + (regRate/m) * nonbiasT3
    theta4Grads[:, 1:theta4Grads.shape[1]] = theta4Grads[:, 1:theta4Grads.shape[1]] + (regRate/m) * nonbiasT4
    
    
    return J,theta1Grads,theta2Grads,theta3Grads,theta4Grads
    
    
def trainNetwork(X, Y, theta1, theta2, theta3, theta4, regRate, learnRate, iters):
    
    for i in range (0,iters):
        #calculate gradients
        startTime = time.time()
        
        cost, theta1Grads, theta2Grads, theta3Grads, theta4Grads = costJ(X, Y, theta1, theta2, theta3, theta4, regRate)
        
        #update thetas
        theta1 = theta1 - learnRate * theta1Grads
        theta2 = theta2 - learnRate * theta2Grads
        theta3 = theta3 - learnRate * theta3Grads
        theta4 = theta4 - learnRate * theta4Grads
        
        endTime = time.time()
        
        deltaTime = endTime - startTime
        
        print('iteration',i,cost, deltaTime) 
        
    return theta1, theta2, theta3, theta4


def plotLearningRate(X, Y, XCV, YCV, theta1, theta2, theta3, theta4, regRate, learningRate, iters):
    
    m=X.shape[0]
    
    for i in range(1, m):
        XTSub = X[0:i,:]
        YTSub = Y[0:i]
        
        theta1, theta2, theta3, theta4 = trainNetwork(XTSub,YTSub,theta1,theta2,theta3, theta4, regRate, learningRate, iterations)
        costTrain, t, t, t, t = costJ(XTSub,YTSub,theta1,theta2,theta3,theta4,0)
        costCV, t, t, t, t = costJ(XCV,YCV,theta1,theta2,theta3,theta4,0)
        print('i', i, 'T:',costTrain,'CV:', costCV)
#%%    
#plotLearningRate(X, Y, xcv, ycv, theta1, theta2, theta3, theta4, regRate, learningRate, iterations)
#validation curve (try training with different lambdas)
#%% Train
theta1, theta2, theta3, theta4 = trainNetwork(X,Y,theta1,theta2, theta3,theta4,regRate, learningRate, iterations)

def saveThetaMatrix(t, name):
    pd.DataFrame(t).to_csv(name, index = False)
    
saveThetaMatrix(theta1, 't1')
saveThetaMatrix(theta2, 't2')
saveThetaMatrix(theta3, 't3')
saveThetaMatrix(theta4, 't4')

#%% Output predictions
preds = predict(predata, theta1, theta2, theta3, theta4 )

imgNum = np.zeros((preds.size,1), dtype=int)


for i in range(0, imgNum.shape[0]):
    imgNum[i] += i+1

preds = np.reshape(preds, (preds.size,1))

preds = pd.DataFrame(np.append(imgNum,preds, axis = 1))



#%%
preds.columns = ['ImageId','Label']
preds.to_csv('results.csv', index = False)
