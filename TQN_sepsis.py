#!/usr/bin/env python
# coding: utf-8

# author: Yeo Jin Kim
# date: 03/05/2020
# File: Induce and evaluate the action policy 
#       using the time-aware RL methods 
#       for septic treatment

import numpy as np
import pandas as pd
import argparse
import os
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
    pid = 'VisitIdentifier'
    label = 'Shock'
    timeFeat = 'MinutesFromArrival'
    discountFeat = 'DynamicDiscount' 
    rewardFeat = 'reward'


    date = ''
    polTDmode = True  ####### True when using the time difference 'TD' for policy induction
        
    actions = [i for i in range(4)] 
    Qfeat = ['Q'+str(i) for i in actions]
    numFeat= ['HeartRate', 'RespiratoryRate','PulseOx', 'SystolicBP', 'DiastolicBP', 'MAP', 'Temperature', 'Bands',
               'BUN', 'Lactate', 'Platelet', 'Creatinine', 'BiliRubin','WBC', 'FIO2']
    if polTDmode:
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
        
        self.load_model = True
        self.date = str(datetime.datetime.now().strftime('%m%d%H'))

        self.save_results = True
        self.func_approx = 'LSTM' # 'LSTM' for RQN or 'FC' for DQN
        self.batch_size = 32
        self.period_save = 10000
        self.period_eval = 10000
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
    
    pid = 'VisitIdentifier'
    timeFeat = 'MinutesFromArrival'
    label = 'Shock'
    discountFeat = 'DynamicDiscount'
    date = ''
        
    actions = [i for i in range(4)]
    
    idFeat = [pid, timeFeat]
    numFeat= ['HeartRate', 'RespiratoryRate','PulseOx', 'SystolicBP', 'DiastolicBP', 'MAP', 'Temperature', 'Bands',
               'BUN', 'Lactate', 'Platelet', 'Creatinine', 'BiliRubin','WBC', 'FIO2']
        
    actFeat = ['a_'+str(i) for i in range(4)]
    oaFeat = ['oa_'+str(i) for i in range(4)]
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
    #print("nextNumFeat: ", env.nextNumFeat)   
    #print("nextstateFeat: ", env.nextStateFeat)

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
     
    print("static gamma: {}".format(env.gamma))
    
    df.loc[:, env.discountFeat] = (env.belief**(df['TD']/focusTimeWindow)).values.tolist()
    
    print("mean TDD (before set fillna(0)): {:.4f}".format(df[env.discountFeat].mean()), end=' ')
    #df[decayfeat] *= df.groupby(pid).shift(1)[decayfeat].fillna(1) # fillna(1): keep tgamma for the first state
    df.loc[:, env.discountFeat] = df[env.discountFeat].fillna(0).tolist() # 0 for the terminal state (at the end of trajectory)
    print(" (after): {:.4f}".format(df[env.discountFeat].mean()))
    if outfile != '':
        df.to_csv(outfile, index=False)
    return df

# Set flag for splitter of numerical feature
def setData(env, df, val_df, test_df, hdf):
    splits = []   
    if 'pred' in env.splitter:
        df, hdf, posdf, negdf = cq.setPredFlag(df, env.pred_basis, env.apx, hdf)
        test_df, hdf, pos_testdf, neg_testdf = cq.setPredFlag(test_df, env.pred_basis, env.apx, hdf)
        splits.append('PredFlag')
   
    # Temporal discount
    if 'Expo' in env.character:
        # Discount: calculate the corresponding discount with belief and target time   
        avgTimeItv = df[df.TD!=0].TD.mean()
        df = setExponentialDiscount(env, df, env.belief, env.targetFuture, avgTimeItv, outfile='')
        val_df = setExponentialDiscount(env, val_df, env.belief, env.targetFuture, avgTimeItv, outfile='')
        test_df = setExponentialDiscount(env, test_df, env.belief, env.targetFuture, avgTimeItv, outfile='')

       
    splitTrain, hdf, streamNum = cq.splitData(df, splits, hdf)
    splitVal, hdf, _ = cq.splitData(val_df, splits, hdf)
    
    if env.load_data == 'iter': 
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

def setDEBUG(df, val_df, test_df, env):
    print("unique actions - train: {}, val: {}, test: {}".format(df.Action.unique(), val_df.Action.unique(), \
          test_df.Action.unique()))

    df = df[df[env.pid].isin(df[env.pid].unique()[:20])]
    val_df = val_df[val_df[env.pid].isin(val_df[env.pid].unique()[:5])]
    test_df = test_df[test_df[env.pid].isin(test_df[env.pid].unique()[:10])]
    env.numSteps = 400
    env.period_save = 1
    env.period_eval = 100
    env.kfold = 2
    print("polStateFeat: {} - {}".format(len(env.stateFeat), env.stateFeat))
    print("trainX: {} - {}".format(np.shape(trainX), trainX[:3]))
    print("valX: {} - {}".format(np.shape(valX), trainX[:3]))
    print("train-reward: {}".format(df.reward.describe()))
    print("val-reward: {}".format(val_df.reward.describe()))        
    return df, val_df, test_df, env


# Changes:
# 101319: 
# 1) iteration-based LSTM data generation
# 2) val_df: 100 samples from training

if __name__ == '__main__':
    
    simMaxSeqLen = 5
    parser = argparse.ArgumentParser()
    env = parsing(parser)
    hdf = cq.saveHyperParameters(env) 
    pdqndir = "../data/preproc_pdqn/"
    model_dir = '' 
    path = '../Sepsis/data/'
    
    # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = env.gpuID  # GPU-ID "0" or "0, 1" for multiple
    env.config = tf.ConfigProto()
    env.config.gpu_options.per_process_gpu_memory_fraction = 0.05
    session = tf.Session(config=env.config) 
    startTime = time.time()

    random.seed(env.cvFold+1) 
#     usecols = ['VisitIdentifier','MinutesFromArrival', 'TD', 'HeartRate', 'RespiratoryRate', 'PulseOx', \
#            'OxygenSource', 'OxygenFlow', 'FIO2', 'SystolicBP', 'DiastolicBP', 'Temperature', 'MAP', 'Bands', \
#            'BUN', 'Lactate', 'Platelet', 'Creatinine', 'BiliRubin', 'WBC', 'Procalcitonin', 'CReactiveProtein', \
#            'SedRate',  'reward', 'Action', 'a_0', 'a_1', 'a_2', 'a_3', 'oa_0', 'oa_1', 'oa_2', 'oa_3']
#     feat_org = [f+'_org' for f in feat]
    feat_next = ['next_' + f for f in env.numFeat]
    featQ = ['Q'+str(i) for i in range(4)]

    usecols = ['VisitIdentifier', 'MinutesFromArrival']+env.numFeat+feat_next+featQ+['Shock','Action', 'reward',
                                                                                           'next_action', 'done'] #'utility',
    if env.polTDmode==False:
        usecols += ['TD']


    if 'mayo' in env.load_data: # used Expert reward
        df, val_df, env = cq.getData(env, '../Sepsis/data/', "mayo_sars60_train_1217", "mayo_sars60_test_1217") 
        df = df[usecols]
        val_df = val_df[usecols]
        if env.DEBUG:
            print("feat_next: ", df.loc[:2, feat_next])
    else:
        print("No data loaded")

    test_df = val_df
       
    # Load LSTM data                 
    if 'LSTM' in env.character:
        print("load LSTM data")
        
        if env.load_data == 'iter':  # when making LSTM data every iteration
            trainX, trainX_next, valX, valX_next = [], [], [], []    

        elif 'mayo_TD' in env.load_data: # 16 featurees, including 'TD'
            with open (path+'lstm/mayo_TD_sars60_msl5_TD_msl5_trainX_current_0108_t5.pk', 'rb') as fp:
                trainX = pickle.load(fp)
            with open (path+'lstm/mayo_TD_sars60_msl5_TD_msl5_trainX_next_0108_t5.pk', 'rb') as fp:
                trainX_next = pickle.load(fp)
            with open (path+'lstm/mayo_TD_sars60_msl5_TD_msl5_testX_current_0108_t5.pk', 'rb') as fp:
                valX = pickle.load(fp)
            with open (path+'lstm/mayo_TD_sars60_msl5_TD_msl5_testX_next_0108_t5.pk', 'rb') as fp:
                valX_next = pickle.load(fp)
            
                
        elif 'mayo' in env.load_data: # 15 features
            with open (path+'lstm/mayo_sars60_msl5_trainX_current_1217_t5.pk', 'rb') as fp:
                trainX = pickle.load(fp)
            with open (path+'lstm/mayo_sars60_msl5_trainX_next_1217_t5.pk', 'rb') as fp:
                trainX_next = pickle.load(fp)
            with open (path+'lstm/mayo_sars60_msl5_testX_current_1217_t5.pk', 'rb') as fp:
                valX = pickle.load(fp)
            with open (path+'lstm/mayo_sars60_msl5_testX_next_1217_t5.pk', 'rb') as fp:
                valX_next = pickle.load(fp)
            
            
        if env.DEBUG:
            print("LSTM - trainX: {}, valX: {}".format(np.shape(trainX), np.shape(valX))) 
            print("trainX: ", trainX[:1])
            print("trainX_next: ", trainX_next[:1])
            print("testX: ", valX[:1])            
            print("testX_next: ", valX_next[:1])
                
        
    print("LSTM shape: train {}, val {}".format( trainX.shape, valX.shape))
            
    if env.DEBUG:
        df, val_df, test_df, env = setDEBUG(df, val_df, test_df, env)
    
    simEnv = SimEnvironment(keyword='simulator', hidden_size=64, maxSeqLen=simMaxSeqLen,\
                            simulator = '', simulatorName = 'simulator', policy = test_df, \
                            policyName = env.filename, policy_sess = '', splits = [])
  
    df = cq.initPER(df, env)
    
    env.train_posvids, env.train_negvids, env.train_totvids = lp.statLabels(df, env.label, '', env.pid)
    env.test_posvids, env.test_negvids, env.test_totvids = lp.statLabels(test_df, env.label, '', env.pid)

    # Set data
    #  if 'TD' not in df.columns.tolist():
    #      df['TD'] = (df.groupby(env.pid).shift(-1)[env.timeFeat] - df[env.timeFeat]).fillna(0).values
        
    avgTD = df[df.TD!=0].TD.mean()
    env.gamma = (env.belief)**(avgTD/env.targetFuture)
    print("belief: {}, targetFuture: {} min, avgTD: {:.4f}, discount: {}".format(env.belief, env.targetFuture, avgTD, env.gamma))
    print("State feat: {}".format(env.stateFeat)) 
    
    hdf, data, df, val_df, test_df, env.streamNum = setData(env, df, val_df, test_df, hdf)


        
    runTime = []
    for i in range(env.cvFold,env.cvFold+1):#env.repeat):
        print(" ******** ", i, " *********") 
        print("streamNum: {}".format(env.streamNum))
        startTime = time.time()
        env.fold = i
        #(tf, env, simEnv, df, val_df, test_df, hdf, data, saveFolder, model_dir)
        save_dir = cq.RLprocess(tf, env, simEnv, df, val_df, test_df, hdf, data, saveFolder='res_sepsis/', model_dir = model_dir)    
        curRunTime = (time.time()-startTime)/60
        runTime.append(curRunTime)
        print("Learning Time: {:.2f} min".format(curRunTime))
        hdf.loc[len(hdf)] = ['learning_time', curRunTime]
        hdf.to_csv(save_dir+'hyper_parameters.csv', index=False)
    print("Learning Time: {}".format(runTime))            
    print("Avg. Learning Time: {:.2f} min".format(np.mean(runTime)))     


# python CRQN_sepsis_mayo.py -r DR -a mayo_TD_sars60_15feat_exRwd -s none -k lstm -c LSTM  -b 0.1 -tf 2880 -d 0.97 -hu 128 -t 1000000 -rp 1 -msl 5 -g 3 -cvf 0
# python CRQN_sepsis_mayo.py -r DR -a mayo_TD_sars60_15feat_exRwd -s none -k lstm -c LSTM_Expo  -b 0.1 -tf 2880 -d 0.97 -hu 128 -t 1000000 -rp 1 -msl 5 -g 3 -cvf 0   

