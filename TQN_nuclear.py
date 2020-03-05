#!/usr/bin/env python
# coding: utf-8

# author: Yeo Jin Kim
# date: 03/05/2020
# File: Induce and evaluate the action policy 
#       using the time-aware RL methods 
#       for nuclear reactor control

import numpy as np
import pandas as pd
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
import datetime
import time
import tensorflow as tf
import pickle

import TQN as cq
import lib_preproc as lp
import lib_dqn_lstm as ld
import random

class Environment(object):
    config = []
    pid = 'Episode'
    label = 'Unsafe'
    timeFeat = 'time'
    discountFeat = 'DynamicDiscount' 
    rewardFeat = 'reward' #'Reward'
    TDmode = True ####################### CHANGE
    
    date = ''
    actionNum = 33 
    
    actions = [i for i in range(actionNum)] 
    Qfeat = ['Q'+str(i) for i in actions]
    
    numFeat = ['FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8', 'TL9', 'TL14', 'PS1', 'PS2', 'PH1', 'PH2', 'cv42C','cv43C'] 
    if TDmode:
        numFeat += ['TD'] 
        
    nextNumFeat = [f + '_next' for f in numFeat]
    stateFeat = numFeat #[]
    nextStateFeat = [f + '_next' for f in stateFeat]
        
    train_posvids = []
    train_negvids = [] 
    train_totvids = []
    test_posvids = []
    test_negvids = [] 
    test_totvids = []
    
    def __init__(self, args):
        self.rewardType = args.r
        self.keyword = args.k
        self.load_data = args.a
        self.character = args.c
        self.gamma = float(args.d)
        self.splitter = args.s
        self.streamNum = 0
        self.LEARNING_RATE = True # use it or not
        self.learnRate = 0.0001 # init_value (αk+1 = 0.98αk)
        self.learnRateFactor = 0.98
        self.learnRatePeriod = 5000
        self.belief = float(args.b)
        self.targetFuture = float(args.tf)
        self.hidden_size = int(args.hu)
        self.numSteps = int(args.t)
        self.discountFeat = args.df
        self.pred_basis = 0
        self.gpuID = str(args.g)
        self.apx = 0 # float(args.apx)
        self.repeat = int(args.rp)
        self.maxSeqLen = int(args.msl)

        self.per_flag = True
        self.per_alpha = 0.6 # PER hyperparameter
        self.per_epsilon = 0.01 # PER hyperparameter
        self.beta_start = 0.9 # the lower, the more prioritized
        self.reg_lambda = 5
        self.Q_clipping = False # for Q-value clipping 
        self.Q_THRESHOLD = 1000 # for Q-value clipping
        self.REWARD_THRESHOLD = 1000
        self.tau = 0.001 #Rate to update target network toward primary network
        if 'pred' in self.splitter:
            env.pred_basis = float(args.pb)
        
        self.pred_res = 0 # inital target prediction result for netowkr training
        self.gamma_rate = 1 # gamma increasing rate (e.g. 1.001)
        
        self.DEBUG = False 
        
        self.load_model = True #True
        self.date = str(datetime.datetime.now().strftime('%m%d%H'))

        self.save_results = True
        self.func_approx = 'LSTM' #'FC_S2' #'FC' 
        self.batch_size = 32
        self.period_save = 20000
        self.period_eval = 20000
        self.saveResultPeriod = 200000
     
        self.cvFold = int(args.cvf)
        self.splitInfo = 'none'
        self.filename = self.splitter+'_'+self.keyword+'_'+self.character +'_b'+str(int(self.belief*10))+ '_g'+ \
                        str(int(self.gamma*100)) +'_h'+str(self.hidden_size)+ '_'+self.load_data 
        if self.DEBUG:
            self.filename = 'DEBUG_' + self.filename
    

class SimEnvironment(object):
    # class attributes
    config = []
    
    pid = 'Episode'
    timeFeat = 'time'
    label = 'Unsafe'
    discountFeat = 'DynamicDiscount'
    date = ''
    actionNum = 33 
    
    actions = [i for i in range(actionNum)]

    
    idFeat = ['Episode', 'time']
    numFeat= ['TD', 'FL1', 'FL6', 'FL19', 'TA21s1', 'TB21s11', 'TL8', 'TL9', 'TL14', 'PS1', 'PS2', 'PH1', 'PH2', 'cv42C','cv43C'] 
        
    actFeat = ['a_'+str(i) for i in range(actionNum)]
    oaFeat = ['oa_'+str(i) for i in range(actionNum)]
    totFeat = numFeat + actFeat
    predFeat = []

    labelFeat = numFeat
    labelFeat_min_std = []
    labelFeat_max_std = []

    
    def __init__(self, keyword, hidden_size, maxSeqLen, simulator, simulatorName, policy,  policyName, \
                 policy_sess, splits):
        self.keyword = keyword
        self.hidden_size = hidden_size
        self.maxSeqLen = maxSeqLen

        self.inFeat = self.numFeat
        self.outFeat = self.numFeat
        
        self.simulator = simulator
        self.simulatorName = simulatorName
        self.policy = policy
        self.policyName = policyName

        self.evalSimulator = ''
        self.evalPolicy = ''
    
        self.policy_sess = policy_sess    
        self.splits = splits
        self.pred_model = None
        self.pred_basis = 0
        self.minMeasNum = 3

    
def parsing(parser):
    
    parser.add_argument("-g")   # GPU ID#
    parser.add_argument("-r")   # i: IR or DR
    parser.add_argument("-k")   # keyword for models & results
    parser.add_argument("-msl")  # max sequence length for LSTM
    parser.add_argument("-a")   # 'agg' for aggregation or 'raw' for raw data
    parser.add_argument("-d")   # discount factor gamma
    parser.add_argument("-s")   # splitter: prediction
    parser.add_argument("-apx")   # sampling mode for prediction: approx or not 
    parser.add_argument("-pb") # pred_val basis to distinguish pos from neg (0.5, 0.9, etc.)
    parser.add_argument("-c") # characteristics of model
    parser.add_argument("-l") # learning_rate
    parser.add_argument("-b") # belief for dynamic TDQN
    parser.add_argument("-tf") # target future time
    parser.add_argument("-hu") # hidden_size
    parser.add_argument("-t") # training iteration
    parser.add_argument("-df") # discount feature (ExpTBD or HyperTBD)
    parser.add_argument("-rp") # repeat to build a model
    parser.add_argument("-cvf") # repeat to build a model
    args = parser.parse_args()

    env = Environment(args)
    env.stateFeat = env.numFeat[:]
    
    # update numfeat & state_features
       
    if 'pf' in env.character: # = predFeat
        env.numfeat += ['pred_val']
        env.stateFeat += ['pred_val']
        
    if 'MI' in env.character:
        env.stateFeat += [i+'_mi' for i in env.numfeat]

    env.nextNumFeat = ['next_'+s for s in env.numFeat]
    env.nextStateFeat = ['next_'+s for s in env.stateFeat]


    # update filename    
    if 'pred' in env.splitter:
        env.filename = env.filename + '_pb'+str(int(env.pred_basis*1000))
    
    
    return env

def round_cut(x):
    return int(np.round(x*100))/100

def get_STGamma(belief, focusTimeWindow, avgTimeInterval): # Exponential static temporal discount
    return belief**(avgTimeInterval/focusTimeWindow)

def setExponentialDiscount(env, df, belief, focusTimeWindow, avgTimeInterval, outfile):
    if 'TD' not in df.columns:
        df.loc[:, 'TD'] = (df.groupby(env.pid).shift(-1)[env.timeFeat]-df[env.timeFeat]).fillna(0).values.tolist()
        
    tgamma =  get_STGamma(belief, focusTimeWindow, avgTimeInterval) 
    
    print("static gamma: {}".format(tgamma))
    df.loc[:, env.discountFeat] = (belief**(df['TD']/focusTimeWindow)).values.tolist()
    
    print("mean TDD (before set fillna(0)): {:.4f}".format(df[env.discountFeat].mean()), end=' ')
    
    df.loc[:, env.discountFeat] = df[env.discountFeat].fillna(0).tolist() # 0 for the terminal state (at the end of trajectory)
    print(" (after): {:.4f}".format(df[env.discountFeat].mean()))
    if outfile != '':
        df.to_csv(outfile, index=False)
    return df

# Set flag for splitter of numerical feature
def setData_namac(env, df, val_df, test_df, hdf):
    splits = []   
    if 'pred' in env.splitter:
        df, hdf, posdf, negdf = cq.setPredFlag(df, env.pred_basis, env.apx, hdf)
        test_df, hdf, pos_testdf, neg_testdf = cq.setPredFlag(test_df, env.pred_basis, env.apx, hdf)
        splits.append('PredFlag')
   
    # Dynamic discount
    # targetFuture = np.round(env.targetFuture, 6) 
    if 'Expo' in env.character:
        print("Set the exponential discount") #: {}".format(env.discountFeat))
        avgTimeItv = df[df.TD!=0].TD.mean()
        df = setExponentialDiscount(env, df, env.belief, env.targetFuture, avgTimeItv, outfile='')
        val_df = setExponentialDiscount(env, val_df, env.belief, env.targetFuture, avgTimeItv, outfile='')
        test_df = setExponentialDiscount(env, test_df, env.belief, env.targetFuture, avgTimeItv, outfile='')
        
        
    splitTrain, hdf, streamNum = cq.splitData(df, splits, hdf)
    splitVal, hdf, _ = cq.splitData(val_df, splits, hdf)
    
    if env.load_data == 'q3_iter': 
        data = ([],[],[],[])
    else:
        splitTrainX, splitTrainX_next = cq.splitData_LSTM(splitTrain, trainX, trainX_next)
        splitValX, splitValX_next = cq.splitData_LSTM(splitVal, valX, valX_next)

        #trainWhole for training, testAll by spliting 
        trainAll = cq.process_eval_batch(env, df, df, trainX, trainX_next)
        valAll = []
        for i in range(len(splitVal)):
            gen2 = cq.process_eval_batch(env, val_df, splitVal[i], valX, valX_next)
            valAll.append(gen2)
            
        #print("trainAll: {}, testAll: {}".format(np.shape(trainAll), np.shape(testAll)))
        print("Data Preperation Time: {:.2f} min".format((time.time()-startTime)/60))
        print("splitTrain: {}, splitVal: {}, trainAll: {}, valAll: {}".format(len(splitTrain), len(splitVal), \
                                                                              len(trainAll), len(valAll)))
        data = (splitTrain, splitVal, trainAll, valAll)
    
    df, val_df, test_df = ld.setRewardType(env.rewardType, df, val_df, test_df)
    
    return hdf, data, df, val_df, test_df, streamNum

def setDEBUG(df, val_df,  env):
    print("unique actions - train: {}, val: {}".format(df.Action.unique(), val_df.Action.unique()))

    df = df[df[env.pid].isin(df[env.pid].unique()[:20])]
    val_df = val_df[val_df[env.pid].isin(val_df[env.pid].unique()[:5])]

    env.numSteps = 400
    env.period_save = 1
    env.period_eval = 100
    env.kfold = 2
    print("polStateFeat: {} - {}".format(len(env.stateFeat), env.stateFeat))
    print("trainX: {} - {}".format(np.shape(trainX), trainX[:3]))
    print("valX: {} - {}".format(np.shape(valX), trainX[:3]))
    print("train-reward: {}".format(df.reward.describe()))
    print("val-reward: {}".format(val_df.reward.describe()))        
    return df, val_df, env


if __name__ == '__main__':
    
    date = str(datetime.datetime.now().strftime('%m/%d %H:%M'))
    print("Start experiment: {}".format(date))
    
    simMaxSeqLen = 5  
    parser = argparse.ArgumentParser()
    env = parsing(parser) # ld.parsing(parser): can be replaced
    
    hdf = cq.saveHyperParameters(env) 
    pdqndir = "../NAMAC/data/out/"
    model_dir = '' 
    
    # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                # the IDs match nvidia-smi
    #os.environ["CUDA_VISIBLE_DEVICES"] = env.gpuID  # GPU-ID "0" or "0, 1" for multiple
    env.config = tf.ConfigProto() #tf.compat.v1.ConfigProto()#
    #env.config.gpu_options.per_process_gpu_memory_fraction = 0.05
    session = tf.Session(config=env.config)  #tf.compat.v1.Session(config=env.config)
    startTime = time.time()

    random.seed(env.cvFold+1) 
    feat = ['FL1', 'FL19', 'FL6', 'PH1', 'PH2', 'PS1', 'PS2', 'TA21s1', 'TB21s11', 'TL14', 'TL8', 'TL9', 'cv42C', 'cv43C']
    if env.TDmode:
        feat += ['TD']
        
    feat_org = [f+'_org' for f in feat if f not in ['TD']]
    feat_next = ['next_' + f for f in feat]
    featQ = ['Q'+str(i) for i in range(33)]
    usecols = ['Episode', 'time']+feat+feat_org+feat_next+featQ+['Action', 'reward', 'next_action', 'done']
    if env.TDmode==False:
        usecols += ['TD']
    
    # load data -----------------------------------------------------------
    if 'q1_elapA' in env.load_data:  # for 'q1_elapA_TD', use the same data files
        df, val_df, env = cq.getData(env, pdqndir+'Q1/','2_q1_elapA_Rw631pm5116_sars_train_0121',\
                                     '2_q1_elapA_Rw631pm5116_sars_test_0121') # reward == utility
        df = df[usecols]
        val_df = val_df[usecols]

    elif 'q1_elapA' in env.load_data:  # for 'q1_elapA_TD', use the same data files
        df, val_df, env = cq.getData(env, pdqndir+'Q1/','2_q1_elapA_Rw631pm5116_sars_train_1215',\
                                     '2_q1_elapA_Rw631pm5116_sars_test_1215')

        df = df[usecols]
        val_df = val_df[usecols]
        
    elif 'q1_elap' in env.load_data:
        df, val_df, env = cq.getData(env, pdqndir+'Q1/', '2_q1_rwd3_sars_train_1210', '2_q1_rwd3_sars_test_1210')
    
    # Reward -----------------------------------------------------------
    if False:
        df['reward'] = df['utility'].values
        val_df['reward'] = val_df['utility'].values

    # Discount: calculate the corresponding discount with belief and target time   
    avgTD = df[df.TD!=0].TD.mean()
    env.gamma = (env.belief)**(avgTD/env.targetFuture)
    print("belief: {}, targetFuture: {} sec, avgTD: {}, discount: {}".format(env.belief, env.targetFuture, avgTD, env.gamma)) 
        
    print("Episode - train: {}, val: {}".format(len(df[env.pid].unique()), len(val_df[env.pid].unique()))) 
    print("State feat: {}".format(env.stateFeat))
    
    test_df = val_df.copy(deep=True)
        
       
    # Load LSTM data                 
    if 'LSTM' in env.character:
        print("load LSTM data")
        
        if env.load_data == 'q3_iter':  # when making LSTM data every iteration
            trainX, trainX_next, valX, valX_next = [], [], [], []
        
        elif 'q1_elapA_TD' in env.load_data:
            with open (pdqndir+'Q1/lstm/Q1_elapA_Rw541pm5116_sars_TD_msl5_trainX_current_0108_t5.pk', 'rb') as fp:
                trainX = pickle.load(fp)
            with open (pdqndir+'Q1/lstm/Q1_elapA_Rw541pm5116_sars_TD_msl5_trainX_next_0108_t5.pk', 'rb') as fp:
                trainX_next = pickle.load(fp)
            with open (pdqndir+'Q1/lstm/Q1_elapA_Rw541pm5116_sars_TD_msl5_testX_current_0108_t5.pk', 'rb') as fp: 
                valX = pickle.load(fp)
            with open (pdqndir+'Q1/lstm/Q1_elapA_Rw541pm5116_sars_TD_msl5_testX_next_0108_t5.pk', 'rb') as fp:
                valX_next = pickle.load(fp)
            print("LSTM shape: train {}, val {}", trainX.shape, valX.shape)
            
                
        elif 'q1_elapA' in env.load_data:
            with open (pdqndir+'Q1/lstm/Q1_elapA_rwd3_sars_msl5_trainX_current_1212_t5.pk', 'rb') as fp:
                trainX = pickle.load(fp)
            with open (pdqndir+'Q1/lstm/Q1_elapA_rwd3_sars_msl5_trainX_next_1212_t5.pk', 'rb') as fp:
                trainX_next = pickle.load(fp)
            with open (pdqndir+'Q1/lstm/Q1_elapA_rwd3_sars_msl5_testX_current_1212_t5.pk', 'rb') as fp: 
                valX = pickle.load(fp)
            with open (pdqndir+'Q1/lstm/Q1_elapA_rwd3_sars_msl5_testX_next_1212_t5.pk', 'rb') as fp:
                valX_next = pickle.load(fp)

            
        if env.DEBUG:
            print("LSTM - trainX: {}, valX: {}".format(np.shape(trainX), np.shape(valX)))  
                
            
    if env.DEBUG:
        df, val_df, env = setDEBUG(df, val_df,  env)
  
    
    simEnv = SimEnvironment(keyword='simulator1211', hidden_size=64, maxSeqLen=simMaxSeqLen,\
                            simulator = '', simulatorName = 'simulator1211', policy = val_df, \
                            policyName = env.filename, policy_sess = '', splits = [])
  
    df = cq.initPER(df, env)

    # Set data
    hdf, data, df, val_df, test_df, env.streamNum = setData_namac(env, df, val_df, test_df, hdf)


        
    runTime = []
    for i in range(env.cvFold,env.cvFold+1):#env.repeat):
        print(" ******** ", i, " *********") 
        print("streamNum: {}".format(env.streamNum))
        startTime = time.time()
        env.fold = i
        #(tf, env, simEnv, df, val_df, test_df, hdf, data, saveFolder, model_dir)
        save_dir = cq.RLprocess(tf, env, simEnv, df, val_df, test_df, hdf, data, saveFolder='res_namac/', model_dir = model_dir)    
        curRunTime = (time.time()-startTime)/60
        runTime.append(curRunTime)
        print("Learning Time: {:.2f} min".format(curRunTime))
        hdf.loc[len(hdf)] = ['learning_time', curRunTime]
        hdf.to_csv(save_dir+'hyper_parameters.csv', index=False)
    print("Learning Time: {}".format(runTime))            
    print("Avg. Learning Time: {:.2f} min".format(np.mean(runTime)))     


# python CRQN_namac.py -r DR -a q3_elap_td1_14feat_utilReward -s none -k lstm -c LSTM  -b 0.1 -tf 180 -d 0.98 -hu 128 -t 1000000 -rp 1 -msl 5 -g 3 -cvf 0                                          
# python CRQN_namac.py -r DR -a q3_elap_td1_14feat_utilReward -s none -k lstm -c LSTM_Expo  -b 0.1 -tf 180 -d 0.98 -hu 128 -t 1000000 -rp 1 -msl 5 -g 3 -cvf 0                                          

