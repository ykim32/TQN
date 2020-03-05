import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import numpy as np
import math
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
import copy
import shutil
import argparse
import multiprocessing as mp
import datetime
import time
import tbmlib as tl
import lib_dqn_lstm as ld
import lib_preproc as lp


from functools import reduce
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Save the hyper-parameters
def saveHyperParameters(env):    
    hdf = pd.DataFrame(columns = ['type', 'value'])
    hdf.loc[len(hdf)] = ['load_data', env.load_data]
    hdf.loc[len(hdf)] = ['file', env.filename]
    hdf.loc[len(hdf)] = ['gamma', env.gamma]
    hdf.loc[len(hdf)] = ['gamma_rate', env.gamma_rate]
    hdf.loc[len(hdf)] = ['splitter', env.splitter]
    hdf.loc[len(hdf)] = ['hidden_size', env.hidden_size]
    hdf.loc[len(hdf)] = ['batch_size', env.batch_size]
    hdf.loc[len(hdf)] = ['rewardType', env.rewardType]
    #hdf.loc[len(hdf)] = ['LEARNING_RATE', env.LEARNING_RATE]
    hdf.loc[len(hdf)] = ['learning_rate', env.learnRate]
    hdf.loc[len(hdf)] = ['learning_rate_factor', env.learnRateFactor]
    hdf.loc[len(hdf)] = ['learning_rate_period', env.learnRatePeriod]
    hdf.loc[len(hdf)] = ['Q_clipping', env.Q_clipping]
    hdf.loc[len(hdf)] = ['Q_THRESHOLD', env.Q_THRESHOLD]
    hdf.loc[len(hdf)] = ['keyword', env.keyword]
    hdf.loc[len(hdf)] = ['character', env.character]
    hdf.loc[len(hdf)] = ['belief', env.belief]
    
    hdf.loc[len(hdf)] = ['numFeat', env.numFeat]
    hdf.loc[len(hdf)] = ['stateFeat', env.stateFeat]
    hdf.loc[len(hdf)] = ['actions', env.actions]
    hdf.loc[len(hdf)] = ['per_flag', env.per_flag]
    hdf.loc[len(hdf)] = ['per_alpha', env.per_alpha]
    hdf.loc[len(hdf)] = ['per_epsilon', env.per_epsilon]
    hdf.loc[len(hdf)] = ['beta_start', env.beta_start]
    hdf.loc[len(hdf)] = ['reg_lambda', env.reg_lambda]
    hdf.loc[len(hdf)] = ['hidden_size', env.hidden_size]
    hdf.loc[len(hdf)] = ['training_iteration', env.numSteps]
    return hdf

def setGPU(tf, env):
        # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = env.gpuID  # GPU-ID "0" or "0, 1" for multiple
    env.config = tf.ConfigProto()
    env.config.gpu_options.per_process_gpu_memory_fraction = 0.1
    return env
    
# init for PER important weights and params
# Might be better with optimistic initial values > max reward 
def initPER(df, env):
    df.loc[:, 'prob'] = abs(df[env.rewardFeat])
    temp = 1.0/df['prob']
    temp[temp == float('Inf')] = 1.0
    df.loc[:, 'imp_weight'] = pow((1.0/len(df) * temp), env.beta_start)
    return df


def setHiddenLayer(state, hidden_size, phase, last_layer):
    if last_layer:
        fc = tf.contrib.layers.fully_connected(state, hidden_size, activation_fn=None)
    else:
        fc = tf.contrib.layers.fully_connected(state, hidden_size) 
    fc_bn = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=phase)
    fc_ac = tf.maximum(fc_bn, fc_bn*0.01)
    return fc, fc_bn, fc_ac 

#  Q-network uses Leaky ReLU activation
class RQnetwork():
    def __init__(self, env, myScope):
        self.phase = tf.placeholder(tf.bool)
        self.num_actions = len(env.actions)
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        self.state = tf.placeholder(tf.float32, shape=[None, env.maxSeqLen, len(env.stateFeat)],name="input_state")
        self.hidden_state = tf.placeholder(tf.float32, shape=[None, env.hidden_size],name="hidden_state")
        self.pred_res = tf.placeholder(dtype=tf.int32,shape=[])

        lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(env.hidden_size),rnn.BasicLSTMCell(env.hidden_size)])
        self.state_in = lstm_cell.zero_state(self.batch_size,tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(\
                inputs=self.state, cell=lstm_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope+'_rnn')
        self.rnn_output = tf.unstack(self.rnn, env.maxSeqLen, 1)[-1]
        #self.streamA, self.streamV = tf.split(self.rnn_output, 2, axis=1)
        
        self.fc1, self.fc1_bn, self.fc1_ac = setHiddenLayer(self.rnn_output, env.hidden_size, self.phase, 0)
        self.fc2, self.fc2_bn, self.fc2_ac = setHiddenLayer(self.fc1_ac, env.hidden_size, self.phase, 1)
        
        # advantage and value streams
        self.streamA, self.streamV = tf.split(self.fc2_ac, 2, axis=1)
                           
        self.AW = tf.Variable(tf.random_normal([env.hidden_size//2,self.num_actions])) 
        self.VW = tf.Variable(tf.random_normal([env.hidden_size//2,1]))    
        self.Advantage = tf.matmul(self.streamA, self.AW)     
        self.Value = tf.matmul(self.streamV, self.VW)
        #Then combine them together to get our final Q-values.
        self.q_output = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
       
        self.predict = tf.argmax(self.q_output,1, name='predict') # vector of length batch size
        
        #Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,self.num_actions,dtype=tf.float32)
        
        # Importance sampling weights for PER, used in network update  xdg       
        self.imp_weights = tf.placeholder(shape=[None], dtype=tf.float32)
        
        # select the Q values for the actions that would be selected         
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot), reduction_indices=1) # batch size x 1 vector
        
        # regularisation penalises the network when it produces rewards that are above the
        # reward threshold, to ensure reasonable Q-value predictions      
        self.reg_vector = tf.maximum(tf.abs(self.Q)-env.REWARD_THRESHOLD,0)
        self.reg_term = tf.reduce_sum(self.reg_vector)
        self.abs_error = tf.abs(self.targetQ - self.Q)
        self.td_error = tf.square(self.targetQ - self.Q)
        
        # below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)
        
        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if env.per_flag:
            self.loss = tf.reduce_mean(self.per_error) + env.reg_lambda*self.reg_term
        else:
            self.loss = self.old_loss + env.reg_lambda*self.reg_term
            
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
        self.trainer = tf.train.AdamOptimizer(learning_rate=env.learnRate)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
        # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.trainer.minimize(self.loss)
            
class RQnetwork2():
    def __init__(self, env, myScope): # available_actions, state_features, hidden_size, func_approx,
        self.phase = tf.placeholder(tf.bool)
        self.num_actions = len(env.actions)
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        self.state = tf.placeholder(tf.float32, shape=[None, env.maxSeqLen, len(env.stateFeat)],name="input_state")
        self.hidden_state = tf.placeholder(tf.float32, shape=[None, env.hidden_size],name="hidden_state")
        self.pred_res = tf.placeholder(dtype=tf.int32,shape=[])

        lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(env.hidden_size),rnn.BasicLSTMCell(env.hidden_size)])
        self.state_in = lstm_cell.zero_state(self.batch_size,tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(\
                inputs=self.state, cell=lstm_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope+'_rnn')
        self.rnn_output = tf.unstack(self.rnn, env.maxSeqLen, 1)[-1]
        self.streamA, self.streamV = tf.split(self.rnn_output, 2, axis=1)

        #self.fc_1 = tf.contrib.layers.fully_connected(self.state, hidden_size)#, activation_fn=None
        #self.fc_1_bn = tf.contrib.layers.batch_norm(self.fc_1, center=True, scale=True, is_training=self.phase)
        #self.fc_1_ac = tf.maximum(self.fc_1_bn, self.fc_1_bn*0.01)
        # advantage and value streams
        # self.streamA, self.streamV = tf.split(self.fc_1_ac, 2, axis=1)
        
        # advantage and value streams
        if self.pred_res == 1:
            self.streamAP, self.streamVP = tf.split(self.rnn_output, 2, axis=1)  
            #streamA
            self.fc4_AP, self.fc4_bn_AP, self.fc4_ac_AP = setHiddenLayer(self.streamAP, env.hidden_size//2, self.phase, 0)
            self.fc5_AP, self.fc5_bn_AP, self.streamAP = setHiddenLayer(self.fc4_ac_AP, env.hidden_size//2, self.phase, 1)              
            #streamV
            self.fc4_VP, self.fc4_bn_VP, self.fc4_ac_VP = setHiddenLayer(self.streamVP, env.hidden_size//2, self.phase, 0)
            self.fc5_VP, self.fc5_bn_VP, self.streamVP = setHiddenLayer(self.fc4_ac_VP, env.hidden_size//2, self.phase, 1)
      
            self.AWP = tf.Variable(tf.random_normal([env.hidden_size//2,self.num_actions]))
            self.VWP = tf.Variable(tf.random_normal([env.hidden_size//2,1]))  
            self.Advantage, self.Value = getValues(self.streamAP, self.streamVP, self.AWP, self.VWP)

        else:
            self.streamAN, self.streamVN = tf.split(self.rnn_output, 2, axis=1)
            # streamA
            self.fc4_AN, self.fc4_bn_AN, self.fc4_ac_AN = setHiddenLayer(self.streamAN, env.hidden_size//2, self.phase, 0)
            self.fc5_AN, self.fc5_bn_AN, self.streamAN = setHiddenLayer(self.fc4_ac_AN, env.hidden_size//2, self.phase, 1)
            #streamV
            self.fc4_VN, self.fc4_bn_VN, self.fc4_ac_VN = setHiddenLayer(self.streamVN, env.hidden_size//2, self.phase, 0)
            self.fc5_VN, self.fc5_bn_VN, self.streamVN = setHiddenLayer(self.fc4_ac_VN, env.hidden_size//2, self.phase, 1)
      
            self.AWN = tf.Variable(tf.random_normal([env.hidden_size//2,self.num_actions]))
            self.VWN = tf.Variable(tf.random_normal([env.hidden_size//2,1]))  
            self.Advantage, self.Value = getValues(self.streamAN, self.streamVN, self.AWN, self.VWN)
        
        #Then combine them together to get our final Q-values.
        self.q_output = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
       
        self.predict = tf.argmax(self.q_output,1, name='predict') # vector of length batch size
        
        #Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,self.num_actions,dtype=tf.float32)
        
        # Importance sampling weights for PER, used in network update         
        self.imp_weights = tf.placeholder(shape=[None], dtype=tf.float32)
        
        # select the Q values for the actions that would be selected         
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot), reduction_indices=1) # batch size x 1 vector
        
        # regularisation penalises the network when it produces rewards that are above the
        # reward threshold, to ensure reasonable Q-value predictions      
        self.reg_vector = tf.maximum(tf.abs(self.Q)-env.REWARD_THRESHOLD,0)
        self.reg_term = tf.reduce_sum(self.reg_vector)
        self.abs_error = tf.abs(self.targetQ - self.Q)
        self.td_error = tf.square(self.targetQ - self.Q)
        
        # below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)
        
        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if env.per_flag:
            self.loss = tf.reduce_mean(self.per_error) + env.reg_lambda*self.reg_term
        else:
            self.loss = self.old_loss + env.reg_lambda*self.reg_term
            
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
        self.trainer = tf.train.AdamOptimizer(learning_rate=env.learnRate)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
        # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.trainer.minimize(self.loss)
            
def setHiddenLayer(state, hidden_size, phase, last_layer):
    if last_layer:
        fc = tf.contrib.layers.fully_connected(state, hidden_size, activation_fn=None)
    else:
        fc = tf.contrib.layers.fully_connected(state, hidden_size) 
    fc_bn = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=phase)
    fc_ac = tf.maximum(fc_bn, fc_bn*0.01)
    return fc, fc_bn, fc_ac    

def getValues(streamA, streamV, AW, VW):
    Advantage = tf.matmul(streamA,AW)
    Value = tf.matmul(streamV,VW)
    return Advantage, Value

def getErrors(tf, targetQ, imp_weights, q_output, actions_onehot, REWARD_THRESHOLD, reg_lambda):
    # select the Q values for the actions that would be selected         
    Q = tf.reduce_sum(tf.multiply(q_output, actions_onehot), reduction_indices=1) # batch size x 1 vector
        
    # regularisation penalises the network when it produces rewards that are above the
    # reward threshold, to ensure reasonable Q-value predictions  
    reg_vector = tf.maximum(tf.abs(Q)-REWARD_THRESHOLD,0)
    reg_term = tf.reduce_sum(reg_vector)
    abs_error = tf.abs(targetQ - Q)
    td_error = tf.square(targetQ - Q)
        
    # below is the loss when we are not using PER
    old_loss = tf.reduce_mean(td_error)
        
    # as in the paper, to get PER loss we weight the squared error by the importance weights
    per_error = tf.multiply(td_error, imp_weights)

    # total loss is a sum of PER loss and the regularisation term
    if per_flag:
        loss = tf.reduce_mean(per_error) + reg_lambda*reg_term
    else:
        loss = old_loss + reg_lambda*reg_term
    return imp_weights, Q, reg_vector, reg_term, abs_error, td_error, old_loss, per_error, loss 
 
            

def saveTestResult(env, df, ecr, save_dir, i):
    df.loc[:,'ECR']= np.nan
    df.loc[df.groupby(env.pid).head(1).index, 'ECR'] = ecr
    df, rdf = ld.rl_analysis_pdqn(env, df)
    df.to_csv(save_dir+"results/testdf_t"+str(i+1)+".csv", index=False) #h"+str(env.hidden_size)+"_
    return df, rdf


def setNetwork(tf, env):
    tf.reset_default_graph()

    if env.keyword == 'lstm':
        mainQN = RQnetwork(env, 'main')
        targetQN = RQnetwork(env, 'target')
    elif env.keyword == 'lstm_s2':
        mainQN = RQnetwork2(env, 'main')
        targetQN = RQnetwork2(env, 'target')

    saver = tf.train.Saver(tf.global_variables())#, max_to_keep= None)
    init = tf.global_variables_initializer()
    return mainQN, targetQN, saver, init    

def load_RLmodel(sess, tf, save_dir):
    startTime = time.time()
    try: # load RL model
        restorer = tf.train.import_meta_graph(save_dir + 'ckpt.meta')
        restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
        print ("Model restoring time: {:.2f} sec".format((time.time()-startTime)))
    except IOError:
        print ("Error: No previous model found!") 

def rl_run(tf, env, test_df, save_dir):
    tf.reset_default_graph()
    
    if env.keyword == 'lstm':
        mainQN = RQnetwork(env, 'main')
        targetQN = RQnetwork(env, 'target')
    elif env.keyword == 'lstm_s2':
        mainQN = RQnetwork2(env, 'main')
        targetQN = RQnetwork2(env, 'target')

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    with tf.Session(config=env.config) as sess:
        try: # load RL model
            restorer = tf.train.import_meta_graph(save_dir + 'ckpt.meta')
            restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
            print ("Model restored")
        except IOError:
            print ("Error: No previous model found!")
            return
                       
        states = np.array(test_df.loc[test_df.Episode==30, env.stateFeat].values.tolist())
        states = states.reshape(1, states.shape[0], states.shape[1])

        actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state:states, \
                         mainQN.phase:0, mainQN.pred_res:env.pred_res, mainQN.batch_size:env.batch_size}) 
                
        return actions_from_q1
    
def getTargetQ(env, Q2, done_flags, actions_from_q1, rewards, tGammas):
    end_multiplier = 1 - done_flags # handles the case when a trajectory is finished
            
    # target Q value using Q values from target, and actions from main
    double_q_value = Q2[range(len(Q2)), actions_from_q1]

    # empirical hack to make the Q values never exceed the threshold - helps learning
    if env.Q_clipping:
        double_q_value[double_q_value > Q_THRESHOLD] = Q_THRESHOLD
        double_q_value[double_q_value < -Q_THRESHOLD] = -Q_THRESHOLD
    #print("double_q_value:", double_q_value)

    # definition of target Q
    if 'Expo' in env.character or 'Hyper' in env.character: # or 'TBD' in env.character:
        targetQ = rewards + (tGammas*double_q_value * end_multiplier)
    else:
        targetQ = rewards + (env.gamma*double_q_value * end_multiplier)    
    #print("targetQ:", targetQ)
    return targetQ


def process_train_batch_iteration(env, df):#splitTrain, trainAll):  
    
    if env.per_flag: # uses prioritised exp replay
        weights = df['prob'] #splitTrain[env.pred_res]['prob'] # 'prob'이 업데이트: deep copy 아니라 연동됨.
    else:
        weights = None

    pooldf = df #splitTrain[env.pred_res]
    a = pooldf.sample(n=env.batch_size, weights=weights) 
    idx = a.index.values.tolist()
#     if env.DEBUG:
#         print("\n*** batch idx: {}".format(idx))

    actions = np.array(a.Action.tolist())
    next_actions = np.array(a.next_action.tolist()) 
    rewards = np.array(a[env.rewardFeat].tolist())
    done_flags = np.array(a.done.tolist())
    
    if env.maxSeqLen > 1: # LSTM
        states = np.array(ld.makeX_event_given_batch_event(pooldf, a, env.stateFeat, env.pid, env.maxSeqLen))#X[idx] #np.array(ld.makeX_event_given_batch(df, a, state_features, 'VisitIdentifier', timesteps))
        next_states = np.array(ld.makeX_event_given_batch_event(df, a, env.nextStateFeat, env.pid, env.maxSeqLen))#X_next[idx] #np.array(ld.makeX_event_given_batch(df, a, next_states_feat, 'VisitIdentifier', timesteps))
    else:
        states = np.array(a.loc[:, env.stateFeat].values.tolist())
        next_states =  np.array(a.loc[:, env.nextStateFeat].values.tolist())
    
    if 'Expo' in env.character or 'Hyper' in env.character:
        tGammas = np.array(a.loc[:, env.discountFeat].tolist())
    else:
        tGammas = []

#     if env.DEBUG:
#         print("states: {}".format(np.shape(states)))
#         print("next_states:{}".format(np.shape(next_states)))

    return states, actions, rewards, next_states, next_actions, done_flags, tGammas, a


def process_train_batch(env, splitTrain, trainAll):#splitTrain, trainAll):  
    
    if env.per_flag: # uses prioritised exp replay
        weights = splitTrain[env.pred_res]['prob'] # 'prob'이 업데이트 안되는 중. 실제 df만업데이트 되는 거 아님? 아니, deep copy 아니라 연동됨.
    else:
        weights = None
            
    a = splitTrain[env.pred_res].sample(n=env.batch_size, weights=weights) 
    idx = a.index.values.tolist()
        
    states, actions, rewards, next_states, next_actions, done_flags, tGammas, _ = trainAll# 
    states =  states[idx]
    actions = actions[idx] 
    next_actions = next_actions[idx]
    rewards = rewards[idx]
    next_states = next_states[idx]
    done_flags = done_flags[idx]
    if tGammas != []:
        tGammas = tGammas[idx]
    return states, actions, rewards, next_states, next_actions, done_flags, tGammas, a


def process_eval_batch(env, df, data, X, X_next):
    #pool = pd.DataFrame(columns = ['states', 'actions', 'rewards', 'next_states', 'next_actions', 'done_flags', 'tGammas'])
    a = data.copy(deep=True)
    idx = a.index.values.tolist()
    actions = np.array(a.Action.tolist())
    next_actions = np.array(a.next_action.tolist()) 
    rewards = np.array(a[env.rewardFeat].tolist())
    done_flags = np.array(a.done.tolist())
    
    if env.maxSeqLen > 1: # LSTM
        states = np.array(ld.makeX_event_given_batch(df, a, env.stateFeat, env.pid, env.maxSeqLen))#X[idx] #np.array(ld.makeX_event_given_batch(df, a, state_features, 'VisitIdentifier', timesteps))
        next_states = np.array(ld.makeX_event_given_batch(df, a, env.nextStateFeat, env.pid, env.maxSeqLen))#X_next[idx] #np.array(ld.makeX_event_given_batch(df, a, next_states_feat, 'VisitIdentifier', timesteps))
    else:
        states = np.array(a.loc[:, env.stateFeat].values.tolist())
        next_states =  np.array(a.loc[:, env.nextStateFeat].values.tolist())
    
    if 'Expo' in env.character or 'Hyper' in env.character:
        tGammas = np.array(a.loc[:, env.discountFeat].tolist())
    else:
        tGammas = []
    return (states, actions, rewards, next_states, next_actions, done_flags, tGammas, a)   



def rl_learning(tf, env, simEnv, df, val_df, test_df, save_dir, data):
    
    diffActionNum = 0
    diffActionRate = 0
    maxECR = 0
    bestECRepoch = 1
    bestShockRate = 1.0
    bestEpoch = 1
    notImproved = 0
    startTime = time.time()
    learnedFeats = ['target_action'] #'RecordID','target_q'
    save_path = save_dir+"ckpt"#The path to save our model to.

    # The main training loop is here
    tf.reset_default_graph()
    
    if env.keyword == 'lstm':
        mainQN = RQnetwork(env, 'main')
        targetQN = RQnetwork(env, 'target')
    elif env.keyword == 'lstm_s2':
        mainQN = RQnetwork2(env, 'main')
        targetQN = RQnetwork2(env, 'target')

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    target_ops = ld.update_target_graph(trainables, env.tau)

    #with tf.Session(config=env.config) as sess:
    policySess = tf.Session(config=env.config) 
    
    df, env, log_df, maxECR, bestShockRate, bestECRepoch, bestEpoch, startIter = ld.initialize_model(env, policySess, save_dir,\
                                                                           df, save_path, init) # load a model if it exists

    net_loss = 0.0

    splitTrain, splitVal, trainAll, valAll = data #splitTest, testAll  
    
    for i in range(startIter, env.numSteps):
        
        states, actions, rewards, next_states, next_actions, done_flags, tGammas, sampled_df = \
                            process_train_batch(env, splitTrain, trainAll)

        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = policySess.run(targetQN.q_output,feed_dict={targetQN.state:next_states, targetQN.phase:1,\
                    mainQN.pred_res:env.pred_res, mainQN.learning_rate:env.learnRate, targetQN.batch_size:env.batch_size})
        
        if 'sarsa' in env.character:
            targetQ = getTargetQ(env, Q2, done_flags, next_actions, rewards, tGammas) 
        else:
            # Run PDQN according to the prediction
            actions_from_q1 = policySess.run(mainQN.predict,feed_dict={mainQN.state:next_states, \
                         mainQN.phase:1, mainQN.pred_res:env.pred_res, mainQN.learning_rate: env.learnRate, \
                                                         mainQN.batch_size:env.batch_size})  
            targetQ = getTargetQ(env, Q2, done_flags, actions_from_q1, rewards, tGammas)  # This reward ia a state reward?    
            
        # Calculate the importance sampling weights for PER
        imp_sampling_weights = np.array(sampled_df['imp_weight'] / float(max(df['imp_weight'])))
        imp_sampling_weights[np.isnan(imp_sampling_weights)] = 1
        imp_sampling_weights[imp_sampling_weights <= 0.001] = 0.001

        # Train with the batch
        _, loss, error = policySess.run([mainQN.update_model, mainQN.loss, mainQN.abs_error], \
                                  feed_dict={mainQN.state: states, mainQN.targetQ: targetQ, mainQN.actions: actions, \
                                             mainQN.phase: True, mainQN.imp_weights: imp_sampling_weights, \
                                             mainQN.batch_size:env.batch_size})

        # Update target towards main network
        ld.update_target(target_ops, policySess)
        net_loss += sum(error)

        # Set the selection weight/prob to the abs prediction error and update the importance sampling weight
        new_weights = pow((error + env.per_epsilon), env.per_alpha)
        df.loc[df.index.isin(sampled_df.index), 'prob'] = new_weights
        df.loc[df.index.isin(sampled_df.index), 'imp_weight'] = pow(((1.0/len(df)) * (1.0/new_weights)), env.beta_start)

        #run an evaluation on the validation set
        if ((i+1) % env.period_eval==0) or i == 0: # should evaluate the first iteration to check the initial condition 
            saver.save(policySess,save_path)
            av_loss = net_loss/(env.period_save * env.batch_size)          
            net_loss = 0.0
            #print ("Saving PER and importance weights")
            with open(save_dir + 'per_weights.p', 'wb') as f:
                pickle.dump(df['prob'], f)
            with open(save_dir + 'imp_weights.p', 'wb') as f:
                pickle.dump(df['imp_weight'], f)

            #1. Validation: ECR
            if env.DEBUG:
                print("DEBUG: Validation")
            val_df, ecr = ld.do_eval_pdqn_lstm(policySess, env, mainQN, targetQN, val_df, valAll) 

            mean_abs_error = np.mean(val_df.error)
            mean_ecr = np.mean(ecr)
            avg_maxQ = val_df.groupby(env.pid).target_q.mean().mean() # mean maxQ by trajectory 
 
            if i+1 > env.period_eval and i > startIter+env.period_eval:
                curActions = val_df[['target_action']].copy(deep=True)
                diffActionNum = len(val_df[curActions['target_action'] != predActions['target_action']])
                diffActionRate = diffActionNum/len(curActions)

                if env.DEBUG:
                     print("diff: {}".format(val_df[curActions['target_action'] != predActions['target_action']].index))
            predActions = val_df[['target_action']].copy(deep=True)

            simPolicy = [] 
            shockNum, shockRate,vidNum = 0,0,0
            
            if mean_ecr > maxECR and i > 1 :
                maxECR = mean_ecr
                bestECRepoch = i+1
                
            if shockRate < bestShockRate and i>1: 
                bestShockRate = shockRate
                notImproved = 0
                bestEpoch = i+1
                saveModel(save_dir, bestPath = save_dir+'models/regul'+str(i+1)+'/')
                #val_df, rdf = saveTestResult(env, val_df, ecr, save_dir, i)
            else:
                notImproved +=1
                saveModel(save_dir, bestPath = save_dir+'models/regul'+str(i+1)+'/')
                #val_df, rdf = saveTestResult(env, val_df, ecr, save_dir, i)
                
            print("{}/{}/{}/h{}/g{}[{:.0f}] L:{:.2f}, Q:{:.2f}, E:{:.2f} (best: {:.2f} - {}),".
                  format(env.date, env.cvFold, env.character, env.hidden_size,\
                                        env.gamma,(i+1),av_loss, avg_maxQ, mean_ecr, maxECR, bestECRepoch),end=' ')
            print("act:{}({:.3f}) run time: {:.1f} m".format(diffActionNum, diffActionRate, (time.time()-startTime)/60))

            startTime = time.time()
            log_df.loc[len(log_df),:] = [i+1, av_loss, mean_abs_error, avg_maxQ, mean_ecr, env.learnRate, env.gamma, \
                                       diffActionNum, diffActionRate, shockNum, shockRate]
            log_df.to_csv(save_dir+"results/log.csv", index=False)

                                
        if env.pred_res != -1:
            if env.pred_res < env.streamNum-1: # change the target prediction result for the next batch
                env.pred_res += 1
            else:
                env.pred_res = 0
        
    if startIter < env.numSteps:    
        saveModel(save_dir, bestPath = save_dir+'regul'+str(i+1)+'/')
        log_df.to_csv(save_dir+"results/log.csv", index=False)

    #Test & Analysis
    shockNum_all = []
    key = simEnv.simulatorName+'_'+ env.filename 
    save_dir = save_dir+'eval/'+key
    print("Policy: {}".format(save_dir))
    
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    policySess.close 
    return df, test_df #, rdf, mainQN

    

def setRanges(env, yhat):
    for i in range(len(env.labelFeat)):
        if (yhat[:, i] < env.labelFeat_min_std[i]) or (yhat[:, i] > env.labelFeat_max_std[i]):
            yhat[:, i] = env.labelFeat_min_std[i]
    return yhat

# Consider TD: Exclude the first TD, which tends to be long. (waiting time)
def applyTD(env, pdf, yhat, j):
    avgTD = ( pdf.loc[1:j,'TD'].median() + pdf.loc[1:j,'TD'].mean() )/ 2
    predicted = yhat[:, :len(env.numFeat)]
    current = np.array(pdf.loc[j, env.numFeat].values.tolist())
    yhat_final = current + (predicted-current)*pdf.loc[j, 'TD']/avgTD
    return yhat_final

def initSimulData(env, pdf):
    pdf.reset_index(inplace=True, drop=True)
    pdf.loc[:, env.actFeat] = pdf.loc[:, env.oaFeat].values.tolist() 
    
    tgTimeIdx = pdf[pdf.timeWindow==1].index.tolist()
    if len(tgTimeIdx) == 1:
        tgTimeIdx = tgTimeIdx*2

    firstMeasIdx = pdf[pd.notnull(pdf.SystolicBP_org)].head(env.minMeasNum).index
    if len(firstMeasIdx) == 0: # in case of minMeasNum == 0 (Use the original dataset)
        firstMeasIdx = [0]
        print("No measurement of systolicBP: vid = {}".format(e))     
        return pdf

    startIdx = np.max([firstMeasIdx[-1],tgTimeIdx[0]]) #gurantee the startIdx covers the minimum number of measurements
    startIdx = pdf[(pdf.index <= startIdx) & pd.notnull(pdf.SystolicBP_org)].tail(1).index[0] # guarntee the startIdx does not cheat the future value by backward filling.
    
    pdf.loc[tgTimeIdx[0], 'timeWindow'] = np.nan
    pdf.loc[startIdx, 'timeWindow'] = 1
    
    #if simulMode == 'policy':
    pdf.loc[:startIdx, 'target_action'] = pdf.loc[:startIdx, 'Action'] # set the original actions out of the window
    pdf.loc[startIdx:, env.actFeat] = 0 # rest the actions to 0 within the window
    pdf.loc[startIdx:, env.miFeat] = 0
    pdf.loc[startIdx:, 'VasoAdmin'] = np.nan
    pdf.loc[startIdx:, 'Anti_infective'] = np.nan
    return pdf, startIdx, tgTimeIdx


def carry_forward (data, hours, carry_list):
    for column in carry_list:
        #data[column+"_inf"] = np.nan
        data.loc[pd.notnull(data[column]), 'Ref_time'] = data.loc[:, 'MinutesFromArrival']
        data.loc[:,[column,'Ref_time']] = data.groupby(['VisitIdentifier'])[[column,'Ref_time']].ffill()
        data.loc[(data.MinutesFromArrival- data.Ref_time).round(6) > 60*hours, column] = np.nan #column+"_inf"
        data.loc[:,'Ref_time'] = np.nan
    return data



def saveModel(save_dir, bestPath):
    if not os.path.exists(bestPath):
        os.makedirs(bestPath)
    
    shutil.copyfile(save_dir+'checkpoint', bestPath+'checkpoint')
    shutil.copyfile(save_dir+'ckpt.data-00000-of-00001', bestPath+'ckpt.data-00000-of-00001')
    shutil.copyfile(save_dir+'ckpt.index', bestPath+'ckpt.index')
    shutil.copyfile(save_dir+'ckpt.meta', bestPath+'ckpt.meta')
    shutil.copyfile(save_dir+'imp_weights.p', bestPath+'imp_weights.p')
    shutil.copyfile(save_dir+'per_weights.p', bestPath+'per_weights.p')
                    

def RLprocess(tf, env, simEnv, df, val_df, test_df, hdf, data, saveFolder, model_dir): 
    if model_dir == '':
        model_dir = saveFolder+env.load_data+'/'+env.date+'/'+str(env.fold)+'/'+env.filename+'/'
        ld.createResultPaths(model_dir, env.date)
        
    hdf.to_csv(model_dir+'hyper_parameters.csv', index=False)
        
    if 'lstm' in env.keyword:   
        env.func_approx = 'LSTM'

    elif 'dqn' in env.keyword:
        env.func_approx = 'FC'

    print("neg pred: train({}), test({})".format(len(df), len(test_df))) 
    _= rl_learning(tf, env, simEnv, df, val_df, test_df, model_dir, data) 
        
    return model_dir
      
    
def RLprocess_for_run(tf, env, df, val_df, test_df, i, hdf, data, model_dir): 
    save_dir=saveFolder+env.load_data+'/'+env.date+'/'+str(i)+'/'+env.filename+'/'
    ld.createResultPaths(save_dir, env.date)
    hdf.to_csv(save_dir+'hyper_parameters.csv', index=False)
        
    if 'lstm' in env.keyword:   
        env.func_approx = 'LSTM'
    elif 'dqn' in env.keyword:
        env.func_approx = 'FC'

    print("neg pred: train({}), test({})".format(len(df), len(test_df))) 
    _= rl_learning(tf, env, df, test_df, test_df, model_dir, data) 



def getNextState(df, env, test_df, state):
    df[env.nextStateFeat] = df[env.stateFeat]
    df[env.nextStateFeat] = df.groupby(env.pid).shift(-1).fillna(0)[env.nextStateFeat]
    test_df[env.nextStateFeat] = test_df[env.stateFeat]
    test_df[env.nextStateFeat] = test_df.groupby(env.pid).shift(-1).fillna(0)[env.nextStateFeat]
    return df, test_df


def normalization(df, test_df, tot_numfeat):
    train_min = df[tot_numfeat].min()
    train_max = df[tot_numfeat].max()
    df.loc[:,tot_numfeat] = (df.loc[:,tot_numfeat] - train_min) / (train_max- train_min)
    test_df.loc[:,tot_numfeat] = (test_df.loc[:,tot_numfeat] - train_min) / (train_max - train_min)   
    return df, test_df

def standardization(df, test_df, tot_numfeat):
    df.loc[:,tot_numfeat] = (df.loc[:,tot_numfeat] - df[tot_numfeat].mean()) / df[tot_numfeat].std()
    test_df.loc[:,tot_numfeat] = (test_df.loc[:,tot_numfeat] - test_df[tot_numfeat].mean()) / test_df[tot_numfeat].std()
    return df, test_df


def getData(env, path, trainfile, testfile):
    df = ld.initData_env(path+trainfile+".csv", env)  
    test_df = ld.initData_env(path+testfile+".csv", env)    

    return df, test_df, env

# using prediction values (Dynamic)
def setPredFlag(env, df, hdf):
    print("pred_basis: {}".format(env.pred_basis))
    posdf = df[df.pred_val >= env.pred_basis-env.apx]
    negdf = df[df.pred_val < env.pred_basis-env.apx]
    df.loc[posdf.index, 'PredFlag'] = 1 
    df.loc[negdf.index, 'PredFlag'] = 0
    info = "Train-pred: pos({}), neg({})".format(len(posdf),len(negdf) )
    hdf.loc[len(hdf)] = ['splitInfo', info] 
    print(info)
    return df, hdf, posdf, negdf
   

#splitTrain, hdf = splitData(df, splits, hdf)
def splitData(df, splits, hdf):
    splitModels = []
    if len(splits) == 1:
        model1 = df[df[splits[0]]==1]
        model2 = df[df[splits[0]]==0]
        models = [model1, model2]
        streamNum = 2
    elif len(splits) == 2:
        model1 = df[(df[splits[0]]==1)&(df[splits[1]]==1)]
        model2 = df[(df[splits[0]]==1)&(df[splits[1]]==0)]
        model3 = df[(df[splits[0]]==0)&(df[splits[1]]==1)]
        model4 = df[(df[splits[0]]==0)&(df[splits[1]]==0)]
        models =  [model1, model2, model3, model4]       
        streamNum = 4
    else: # no split 
        models = [df]
        streamNum = 1
    
    return models, hdf, streamNum


def splitData_LSTM(models, X, X_next):
    splitX = []
    splitX_next = []
    for i in range(len(models)):
        idx = models[i].index.values.tolist()
        print("split idx: {}".format(np.shape(idx)))
        
        splitX.append(X[idx])
        splitX_next.append(X_next[idx])
    return splitX, splitX_next
    