# -*- coding: utf-8 -*-
"""
NN - Number classifier

784+1 -> 100+1 -> 50+1 ->  25+1 -> 10

    100x785  50x76    25x51    10x26
"""
import pandas as pd
import numpy as np
import time

#%%
#Parameters (last layer needs to be 10 and first layer always needs to be 784)
#hidden layer size and amount of hidden layers are variable
sizeL = [784, 100, 50, 25, 10]

numSamples = 42000

regRate = (1/3)
learningRate = .02
iterations = 50000

#%%
#load data

data = pd.read_csv('train.csv', dtype='float64', nrows = numSamples)
predata = pd.read_csv('test.csv', dtype='float64')

predata=predata.values

numSamples = data.shape[0]
trainSamples = int(numSamples * 0.6)
cvSamples = int(numSamples * 0.25)

#%%
#Process Data
X = data.iloc[0:trainSamples, 1:785].values
Y = data.iloc[0:trainSamples, 0].values

xcv = data.iloc[trainSamples:trainSamples+cvSamples, 1:785].values
ycv = data.iloc[trainSamples:trainSamples+cvSamples, 0].values

#normalize data
def normData(X, mean=False, std=False):
    
    if(mean==False and std == False):
        mean = np.mean(X)
        std = np.std(X)
        return ((X-mean) / std), mean, std
    else:
        return ((X-mean) / std)

X, meanX, stdX = normData(X)
xcv = normData(xcv, meanX, stdX)
predata = normData(predata, meanX, stdX)

#%%
#initialize wts

def loadThetaMatrix(fName, dims):
    t = False
    try:
        t = pd.read_csv(fName)
        t = t.values
    except FileNotFoundError:
        return False
    
    if(t.shape != dims):
        return False
    else:
        return t

def randThetaMatrix(dims):
    eps = np.sqrt(6) / np.sqrt(dims[0] + dims[1])
    #range of wts is [-eps, eps]
    theta = np.random.random_sample(dims) * (2*eps) - eps
    return theta

def initializeTheta(lSizes, forceRand = False):
    nLayers = len(sizeL)
    theta = []
    for i in range(nLayers-1):
        
        if(forceRand == False):
            name = "t" + str(i+1) + ".csv"
            dims = (lSizes[i+1], lSizes[i]+1)
            
            #attempts to load file first
            t = loadThetaMatrix(name, dims)
            
            #check to see if file was loaded correctly
            if isinstance(t, bool):
                #if it wasnt gen a new matrix
                t = randThetaMatrix(dims)
            
           
        else:
            t = randThetaMatrix(dims)
            
        theta.append(t)
        
    return theta
                
        
thetas = initializeTheta(sizeL)
#%%
def sigmoid(X, grad=False):
    if grad:
        sigX = sigmoid(X)
        return np.multiply(sigX, 1-sigX)
    else:    
        return 1 / ( 1+ np.exp(-1*X))
#%%
def addBiasColumn(a):
    return np.append(np.ones((a.shape[0], 1)), a, axis=1)

def feedFowardLayer(a, theta, addBias):
    z = a @ theta.T
    a = sigmoid(z)
    if(addBias == True):
        a = addBiasColumn(a)
    return a, z        

def feedForwardNetwork(X, thetas):
    steps = len(thetas)
 
    #add biases to input layer
    a1 = addBiasColumn(X)
    
    #Feed-forward
    aList = [a1]
    zList = [X]
    for i in range(steps):
        
        #dont add bias to result layer
        addBias = True
        if(i == steps-1):
            addBias = False
            
        a, z = feedFowardLayer(aList[i], thetas[i], addBias)  
        aList.append(a)
        zList.append(z) 

    return aList, zList

def predict(X, thetas):
    steps = len(thetas)
    aList = feedForwardNetwork(X, thetas)
    predictions = np.argmax(aList[steps],axis=1)
    
    return predictions
    
def numToLogicalArray(num):
     ans = np.zeros(10)
     ans[num] = 1
     return ans
    
def costJ(X, Y, thetas, regRate):
    #helpful vars
    J = 0.0
    m = X.shape[0]
    lastL = len(thetas)
    
    tGrads = []
    for i in range(lastL):
        tGrads.append(np.zeros(thetas[i].shape))
    
   
    #Feed-forward
    aList, zList = feedForwardNetwork(X, thetas)
    
    #Cost
    for i in range(0,m):
        hx = aList[lastL][i,:]
        
        ans = numToLogicalArray(int(Y[i]))
        
        J = J - np.sum(np.multiply(np.log(hx), ans) + np.multiply(np.log(1-hx), 1-ans))
    
    J = J / m
    
    #cost regularization (j + all nonbias thetas^2 * (regRate/2m))
    
    totalTheta = 0.0
    
    for i in range(lastL):
        nonbiasTheta = thetas[i][:, 1:thetas[i].shape[1]]
        totalTheta += np.sum(np.power(nonbiasTheta,2))
    
    reg = totalTheta * (regRate/(2*m))
    
    J = J + reg
    
    #Gradients-----SPLIT INTO ANOTHER METHOD
    
    #get smaller subset to calculate gradients
    X, Y = randomSample(X, Y, 600)
    m = X.shape[0]
    aList, zList = feedForwardNetwork(X, thetas)
    
    for i in range(0,m):
    
        aLastI = aList[lastL][i]
        
        ans = numToLogicalArray(int(Y[i]))
        
        dLast = aLastI - ans
        dLast = np.atleast_2d(dLast).T #10,null -> 10,1
        
        #final theta layer gradients
        a4i = aList[lastL-1][[i]] #column array
        tGrads[lastL-1] = tGrads[lastL-1] + dLast @ a4i
        
        
        deltas = [dLast]
        
        for l in range(lastL-1):
            zIndex = (lastL-1) - l
            z2i = zList[zIndex][i]
            
            t2Index = (lastL-1) - l
            d2Index = l
            d2 = deltas[d2Index]
            
            itm = thetas[t2Index].T @ d2 #a4 partial deriv
            itm = np.delete(itm, 0) #remove bias deriv
            delta = np.multiply(itm,  sigmoid(z2i, grad=True))
            delta = np.atleast_2d(delta).T
            
            deltas.append(delta)
            
            aIndex = (lastL-2) - l
            a1i = aList[aIndex][[i]]
            
            tIndex = (lastL-2) - l
            dIndex = l + 1
            tGrads[tIndex] = tGrads[tIndex] + deltas[dIndex] @ a1i
            


    #take average
    for i in range(lastL):
        tGrads[i] = tGrads[i] / m
    
    #regularized gradients ( add nonbias thetas * (regRate/m) )
    for i in range(lastL):
        nonbiasTheta = thetas[i][:, 1:thetas[i].shape[1]]
        tGrads[i][:, 1:tGrads[i].shape[1]] = tGrads[i][:, 1:tGrads[i].shape[1]] + (regRate/m) * nonbiasTheta
    
    
    return J, tGrads
    

def randomSample(X, Y, amount):
    #if amount is larger than dataset, use the entire dataset
    if amount > X.shape[0]:
        amount = X.shape[0]
        
    indecies = np.random.randint(0, X.shape[0], size=(amount))
    
    sampleX = X[indecies, :]
    sampleY = Y[indecies]
    
    return sampleX, sampleY
    
    
def trainNetworkOnce(X, Y, thetas, regRate, learnRate, printRes = True):
        startTime = time.time()
        
        cost, tGrads = costJ(X, Y, thetas, regRate)
        
        #update thetas
        for i in range(len(thetas)):
            thetas[i] = thetas[i] - learnRate * tGrads[i]
        
        endTime = time.time()
        
        deltaTime = endTime - startTime
        
        if(printRes):
            print('Cost:', cost, 'Seconds:', deltaTime) 
        
        return thetas
    
    
    
def trainNetwork(X, Y, thetas, regRate, learnRate, iters, printRes = True):
    
    for i in range (0,iters):
        if(printRes):
            print('iteration',i) 
            
        thetas = trainNetworkOnce(X, Y, thetas, regRate, learnRate, printRes)
        
        
    return thetas


def plotLearningRate(X, Y, XCV, YCV, regRate, learningRate, sizeL, iters):
    
    m=X.shape[0]
    
    thetas = initializeTheta(sizeL, forceRand = True)
    
    for i in range(1, m):
        XTSub = X[0:i,:]
        YTSub = Y[0:i]
        
        thetas = trainNetwork(XTSub,YTSub, thetas, learningRate, iterations)
        costTrain, grads = costJ(XTSub,YTSub,thetas,0)
        costCV, grads= costJ(XCV,YCV,thetas,0)
        print('i', i, 'T:',costTrain,'CV:', costCV)

#%%
def saveThetaMatrix(thetas):
    for i in range(len(thetas)):
        name = "t" + str(i+1) + ".csv"  
        pd.DataFrame(thetas[i]).to_csv(name, index = False)

#%% Output predictions

def outputPredictions(predicitonData, thetas, nameFile='results.csv'):
    preds = predict(predicitonData, thetas)
    
    imgNum = np.zeros((preds.size,1), dtype=int)
    
    
    for i in range(0, imgNum.shape[0]):
        imgNum[i] += i+1
    
    preds = np.reshape(preds, (preds.size,1))
    
    preds = pd.DataFrame(np.append(imgNum,preds, axis = 1))
 
    preds.columns = ['ImageId','Label']
    preds.to_csv(nameFile, index = False)



thetas = trainNetwork(X,Y, thetas, regRate, learningRate, iterations)
    
saveThetaMatrix(thetas)

#%%    
#plotLearningRate(X, Y, xcv, ycv, regRate, learningRate, sizeL, iterations)
#validation curve (try training with different lambdas)
