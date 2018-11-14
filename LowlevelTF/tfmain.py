#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle


#%%
#Parameters 

#try multiple values at onece
batchSize = 4000

regRate = 0.0
learningRate = .005
iterations = 50000

kp = .9

#%%
#load data

data = pd.read_csv('train.csv', dtype='float32')
predata = pd.read_csv('test.csv', dtype='float32')

predata=predata.values

numSamples = data.shape[0]
trainSamples = int(numSamples * 0.8)
cvSamples = int(numSamples * 0.2)

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

#%% save data
with open('data.pickle', 'wb') as f:
    datadict = {'X': X, 'Y': Y,
                'xcv': xcv, 'ycv': ycv,
                'predata': predata}
    pickle.dump(datadict, f)

#%% Compute Graph
graph = tf.Graph()

with graph.as_default():
    #load data
    global_step = tf.Variable(0, trainable=False)
    decayLR = learningRate * ((.5) ** (global_step/1000))

    #load data into gpu
    trainData = tf.constant(X, tf.float32)
    trainLabels = tf.one_hot(Y, 10)
    cvData = tf.constant(xcv, tf.float32)
    cvLabels = tf.one_hot(ycv, 10)

    toPredictData = tf.constant(predata, dtype=tf.float32)

    useValidationData = tf.placeholder_with_default(False, shape=[])
    keepRate = tf.placeholder(tf.float32)


    data = tf.cond(useValidationData, lambda: cvData, lambda: trainData)
    labels = tf.cond(useValidationData, lambda: cvLabels, lambda: trainLabels)



    wt1 = tf.Variable(tf.truncated_normal([784, 400], dtype=tf.float32, stddev=.1))
    b1 = tf.Variable(tf.zeros([400], dtype=tf.float32))

    wt2 = tf.Variable(tf.truncated_normal([400, 100], dtype=tf.float32, stddev=.1))
    b2 = tf.Variable(tf.zeros([100], dtype=tf.float32))

    wt3 = tf.Variable(tf.truncated_normal([100, 30], dtype=tf.float32, stddev=.1))
    b3 = tf.Variable(tf.zeros([30], dtype=tf.float32))

    wt4 = tf.Variable(tf.truncated_normal([30, 10], dtype=tf.float32, stddev=.1))
    b4 = tf.Variable(tf.zeros([10], dtype=tf.float32))

    #Feed forward
    l1_logits = tf.matmul(data, wt1) + b1
    l1_a = tf.nn.relu(l1_logits)
    l1_drop = tf.nn.dropout(l1_a, keep_prob=keepRate)

    l2_logits = tf.matmul(l1_drop, wt2) + b2
    l2_a = tf.nn.relu(l2_logits)
    l2_drop = tf.nn.dropout(l2_a, keep_prob=keepRate)

    l3_logits = tf.matmul(l2_drop, wt3) + b3
    l3_a = tf.nn.relu(l3_logits)
    l3_drop = tf.nn.dropout(l3_a, keep_prob=keepRate)
    
    l4_logits = tf.matmul(l3_drop, wt4) + b4

    loss = (tf.reduce_mean
    (tf.nn.softmax_cross_entropy_with_logits_v2(logits=l4_logits, labels=labels))
            + regRate * tf.nn.l2_loss(wt1)
            + regRate * tf.nn.l2_loss(wt2)
            + regRate * tf.nn.l2_loss(wt3)
            + regRate * tf.nn.l2_loss(wt4))

    optimizer = tf.train.AdamOptimizer(learning_rate=decayLR).minimize(loss, global_step=global_step)

    predictions = tf.nn.softmax(l4_logits)

    correct = tf.equal(tf.arg_max(predictions, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


#%%
iterations = 2500
with tf.Session(graph=graph) as sess:

    tf.global_variables_initializer().run()

    train_l = trainLabels.eval()
    cv_l = cvLabels.eval()

    trainTrainDict = {keepRate: 0.82}
    trainEvalDict = {keepRate: 1}
    cvDict = {useValidationData: True, keepRate: 1}

    for i in range(iterations):
        sess.run([optimizer], feed_dict=trainTrainDict)

        if i == 0 or (i % 5 == 0):
            l, tAcc = sess.run([loss, accuracy], feed_dict=trainEvalDict)
            cvAcc = sess.run([accuracy], feed_dict=cvDict)

            print('Iteration: ', i, 
                    '\n\tLoss: ', l,
                    '\n\tTrain Acc: ', tAcc,
                    '\n\tCV Acc: ', cvAcc)