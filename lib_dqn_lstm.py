import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import pickle
import copy
import random
import tbmlib as tl
import lib_preproc as lp
from keras.preprocessing.sequence import pad_sequences
import os 

def initData_env(file, env):
    df = pd.read_csv(file, header=0) 
    df = df.sort_values([env.pid, env.timeFeat])
    df.reset_index(drop=True, inplace=True)
                  
    # set 'done_flag'
    df['done']=0
    df.loc[df.groupby(env.pid).tail(1).index, 'done'] = 1
    # next actions
    df['next_action'] = 0 
    df.loc[:, 'next_action'] = df.groupby(env.pid).Action.shift(-1).fillna(0)
    df['next_action'] = pd.to_numeric(df['next_action'], downcast='integer')
    # df.loc[:, 'next_actions'] = np.array(sdf.groupby('VisitIdentifier').Action.shift(-1).fillna(0), dtype=np.int)

    # next states
    env.nextStateFeat = ['next_'+s for s in env.stateFeat]
    df[env.nextStateFeat] = df.groupby(env.pid).shift(-1).fillna(0)[env.stateFeat]
    
    # action Qs
    Qfeat = ['Q'+str(i) for i in range(len(env.actions))]
    for f in Qfeat:
        df[f] = np.nan
        
    # Temporal Difference for the decay discount factor gamma * exp (-t/tau)
    df.loc[:, 'TD'] = (df.groupby(env.pid)[env.timeFeat].shift(-1) - df[env.timeFeat]).fillna(0).tolist()
    return df


def initData(file, pid, stateFeat, timeFeat, actions):
    df = pd.read_csv(file, header=0) 
    # set 'done_flag'
    df['done']=0
    df.loc[df.groupby(pid).tail(1).index, 'done'] = 1
    # next actions
    df['next_action'] = 0 
    df.loc[:, 'next_action'] = df.groupby(pid).Action.shift(-1).fillna(0)
    df['next_action'] = pd.to_numeric(df['next_action'], downcast='integer')


    # next states
    nextStateFeat = ['next_'+s for s in stateFeat]
    df[nextStateFeat] = df.groupby(pid).shift(-1).fillna(0)[stateFeat]
    
    # action Qs
    Qfeat = ['Q'+str(i) for i in range(len(actions))]
    for f in Qfeat:
        df[f] = np.nan
        
    # Temporal Difference for the decay discount factor gamma * exp (-t/tau)
    df.loc[:, 'TD'] = (df.groupby(pid)[timeFeat].shift(-1) - df[timeFeat]).tolist()
    return df, nextStateFeat


#Make paths for our model and results to be saved in.
def createResultPaths(save_dir, date):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    if not os.path.exists(save_dir+"results"):
        os.mkdir(save_dir+"results")
    if not os.path.exists(save_dir+"best"):
        os.mkdir(save_dir+"best")
        
    if not os.path.exists('results'):
        os.mkdir("results")
    if not os.path.exists('results/'+date):
        os.mkdir('results/'+date)
    
    print(save_dir)

def setRewardType(rewardType, df, valdf, testdf):
    if rewardType == 'IR':
        print("*** Use IR ")
        IRpath = '../inferredReward/results/'
        irTrain = pd.read_csv(IRpath+'train_IR.csv', header=None)
        irTest = pd.read_csv(IRpath+'test_IR.csv', header=None)
        df['reward'] = irTrain
        valdf['reward'] = irTest
        testdf['reward'] = irTest
    else:
        print("*** Use Delayed Rewards")
    return df, valdf, testdf
 
    
def setAgeFlag(df, test_df, age, hdf):
    splitter = 'AgeFlag'
    trainAge_std = 17.452457113348597
    trainAge_mean = 63.8031197301855 
    df[splitter] = 0
    df.loc[df[df['Age'] * trainAge_std + trainAge_mean >= age].index, splitter] = 1
    test_df[splitter] = 0
    test_df.loc[test_df[test_df['Age']*trainAge_std +trainAge_mean >= age].index, splitter] = 1

    train1 = len(df[df[splitter]==1].VisitIdentifier.unique())
    train0 = len(df[df[splitter]==0].VisitIdentifier.unique())
    test1 = len(test_df[test_df[splitter]==1].VisitIdentifier.unique())
    test0 = len(test_df[test_df[splitter]==0].VisitIdentifier.unique())
    info = "AgeFlag:{} Train - 1({}) 0({}) / Test - 1({}) 0({})".format(age, train1, train0, test1, test0)
    print(info)
    hdf.loc[len(hdf)] = ['splitInfo', info]
    #adf_train = pd.read_csv("../data/preproc/3_3_beforeShock_Prediction_Train_0123.csv", header=0)
    #adf_test = pd.read_csv("../data/preproc/3_3_beforeShock_Prediction_Test_0123.csv", header=0)
    #adf_train.groupby(pid).Age.mean().mean()
    #adf_train.groupby(pid).Age.mean().std()
    
    return df, test_df, splitter, hdf


#------------------
# Training

# function is needed to update parameters between main and target network
# tf_vars are the trainable variables to update, and tau is the rate at which to update
# returns tf ops corresponding to the updates

#  Q-network uses Leaky ReLU activation
class Qnetwork():
    def __init__(self, available_actions, state_features, hidden_size, func_approx, myScope):
        self.phase = tf.placeholder(tf.bool)
        self.num_actions = len(available_actions)
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        self.state = tf.placeholder(tf.float32, shape=[None, len(state_features)],name="input_state")
    
        #if func_approx == 'FC':
        # 4 fully-connected layers ---------------------
        self.fc_1 = tf.contrib.layers.fully_connected(self.state, hidden_size, activation_fn=None)
        self.fc_1_bn = tf.contrib.layers.batch_norm(self.fc_1, center=True, scale=True, is_training=self.phase)
        self.fc_1_ac = tf.maximum(self.fc_1_bn, self.fc_1_bn*0.01)
        self.fc_2 = tf.contrib.layers.fully_connected(self.fc_1_ac, hidden_size, activation_fn=None)
        self.fc_2_bn = tf.contrib.layers.batch_norm(self.fc_2, center=True, scale=True, is_training=self.phase)
        self.fc_2_ac = tf.maximum(self.fc_2_bn, self.fc_2_bn*0.01)
        self.fc_3 = tf.contrib.layers.fully_connected(self.fc_2_ac, hidden_size, activation_fn=None)
        self.fc_3_bn = tf.contrib.layers.batch_norm(self.fc_3, center=True, scale=True, is_training=self.phase)
        self.fc_3_ac = tf.maximum(self.fc_3_bn, self.fc_3_bn*0.01)
        self.fc_4 = tf.contrib.layers.fully_connected(self.fc_3_ac, hidden_size, activation_fn=None)
        self.fc_4_bn = tf.contrib.layers.batch_norm(self.fc_4, center=True, scale=True, is_training=self.phase)
        self.fc_4_ac = tf.maximum(self.fc_4_bn, self.fc_4_bn*0.01)

        # advantage and value streams
        # self.streamA, self.streamV = tf.split(self.fc_3_ac, 2, axis=1)
        self.streamA, self.streamV = tf.split(self.fc_4_ac, 2, axis=1)
                    
        self.AW = tf.Variable(tf.random_normal([hidden_size//2,self.num_actions]))
        self.VW = tf.Variable(tf.random_normal([hidden_size//2,1]))    
        self.Advantage = tf.matmul(self.streamA,self.AW)    
        self.Value = tf.matmul(self.streamV,self.VW)
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
        self.reg_vector = tf.maximum(tf.abs(self.Q)-REWARD_THRESHOLD,0)
        self.reg_term = tf.reduce_sum(self.reg_vector)
        self.abs_error = tf.abs(self.targetQ - self.Q)
        self.td_error = tf.square(self.targetQ - self.Q)
        
        # below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)
        
        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if per_flag:
            self.loss = tf.reduce_mean(self.per_error) + reg_lambda*self.reg_term
        else:
            self.loss = self.old_loss + reg_lambda*self.reg_term

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
        # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.trainer.minimize(self.loss)
            
            
def update_target_graph(tf_vars,tau):
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:int(total_vars/2)]):
        op_holder.append(tf_vars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tf_vars[idx+int(total_vars/2)].value())))
    return op_holder

def update_target(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def update_targetupdate_t (op_holder,sess):
    for op in op_holder:
        sess.run(op)
        
        
def initialize_model(env, sess, save_dir, df, save_path, init):
    log_df = pd.DataFrame(columns = ['timestep', 'avgLoss', 'MAE', 'avgMaxQ', 'avgECR', 'learningRate', 'gamma', \
                             'ActDifNum', 'ActDifRatio', 'shockNum', 'shockRate'])  
    maxECR = 0
    bestShockRate = 1
    bestECRepoch = 1
    bestEpoch = 1
    startIter = 0
       
    if env.load_model == True:
        print('Trying to load model...')
        try:
            restorer = tf.train.import_meta_graph(save_path + '.meta')
            restorer.restore(sess, tf.train.latest_checkpoint(save_dir))        
            print ("Model restored")
            
            log_df = pd.read_csv(save_dir+"results/log.csv", header=0)
            env.learnRate = float(log_df.tail(1).learningRate)
            maxECR = float(log_df.avgECR.max())
            bestShockRate = float(log_df.shockRate.min())
            bestECRepoch = int(log_df.loc[log_df.avgECR.idxmax(), 'timestep'])
            bestEpoch = int(log_df.loc[log_df.shockRate.idxmin(), 'timestep'])
            print("Evaluation period: {} epochs".format(env.period_eval))
            print("Previous bestShockRate: {:.3f} (e={}) / maxECR: {:.2f} (e={})".format(bestShockRate, bestEpoch, maxECR,\
                                                                                     bestECRepoch))
            startIter = int(log_df.tail(1).timestep)+1     
        except IOError:
            print ("No previous model found, running default init")
            sess.run(init)
        try:
            per_weights = pickle.load(open( save_dir + "per_weights.p", "rb" ))
            imp_weights = pickle.load(open( save_dir + "imp_weights.p", "rb" ))

            # the PER weights, governing probability of sampling, and importance sampling
            # weights for use in the gradient descent updates
            df['prob'] = per_weights
            df['imp_weight'] = imp_weights
            print ("PER and Importance weights restored")
        except IOError:
            
            print("No PER weights found - default being used for PER and importance sampling")
    else:
        #print("Running default init")
       
        sess.run(init)
    print("Start Interation: {}".format(startIter))
    #print("Model initialization - done")
    return df, env, log_df, maxECR, bestShockRate, bestECRepoch, bestEpoch, startIter
 

# -----------------
# Evaluation
   
# extract chunks of length size from the relevant dataframe, and yield these to the caller
# Note: 
# for evaluation, some portion of val/test set can be evaluated, but  
# For test, all the test set's data (40497 events) should be evaluated. Not just 1000 events from the first visit.

def do_eval(sess, env, mainQN, targetQN, df):
    
    gen = process_eval_batch(env, df, df) 
    all_q_ret = []
    phys_q_ret = []
    actions_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    ecr = []
    error_ret = [] #0
    start_traj = 1
    for b in gen: # b: every event for the whole test set
        states,actions,rewards,next_states, _, done_flags, tGammas, _ = b
        # firstly get the chosen actions at the next timestep
        actions_from_q1 = sess.run(mainQN.predict,feed_dict={mainQN.state:next_states, mainQN.phase:0, mainQN.batch_size:len(states)})
        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = sess.run(targetQN.q_output,feed_dict={targetQN.state:next_states, targetQN.phase:0, targetQN.batch_size:len(next_states)})
        # handles the case when a trajectory is finished
        end_multiplier = 1 - done_flags
        # target Q value using Q values from target, and actions from main
        double_q_value = Q2[range(len(Q2)), actions_from_q1]
        # definition of target Q
        if 'Expo' in env.character or 'Hyper' in env.character: #'TBD' in env.character:
            targetQ = rewards + (tGammas * double_q_value * end_multiplier)            
        else:
            targetQ = rewards + (env.gamma * double_q_value * end_multiplier)

        # get the output q's, actions, and loss
        q_output, actions_taken, abs_error = sess.run([mainQN.q_output,mainQN.predict, mainQN.abs_error], \
            feed_dict={mainQN.state:states, mainQN.targetQ:targetQ, mainQN.actions:env.actions,
                       mainQN.phase:False, mainQN.batch_size:len(states)})

        # return the relevant q values and actions
        phys_q = q_output[range(len(q_output)), actions]
        agent_q = q_output[range(len(q_output)), actions_taken]
        
#       update the return vals
        error_ret.extend(abs_error)
        all_q_ret.extend(q_output)
        phys_q_ret.extend(phys_q)
        actions_ret.extend(actions)        
        agent_q_ret.extend(agent_q)
        actions_taken_ret.extend(actions_taken)
        ecr.append(agent_q[0])
  
    return all_q_ret, phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, error_ret, ecr


def process_eval_batch(env, df, data):
    a = data.copy(deep=True)   
    
    actions = np.squeeze(a.Action.tolist())
    next_actions = np.squeeze(a.next_action.tolist()) 
    rewards = np.squeeze(a[env.rewardFeat].tolist())
    done_flags = np.squeeze(a.done.tolist())
    
    if env.maxSeqLen > 1: # LSTM
        states = makeX_event_given_batch(df, a, env.stateFeat, env.pid, env.maxSeqLen)
        next_states = makeX_event_given_batch(df, a, env.nextStateFeat,  env.pid, env.maxSeqLen)   
    else:
        states = a.loc[:, env.stateFeat].values.tolist() 
        next_states =  a.loc[:, env.nextStateFeat].values.tolist()
    
    tGammas = np.squeeze(a.loc[:, env.discountFeat].tolist())
    yield (states, actions, rewards, next_states, next_actions, done_flags, tGammas, a)

        
def do_eval_pdqn_split(sess, env, mainQN, targetQN, gen, pred_res):
    all_q_ret = []
    phys_q_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    error_ret = []
    
    #for b in gen: # gen: a set of every event (b) with same pred_res (not visit) 
    #    print("b", np.shape(b))
    states, actions, rewards, next_states, _, done_flags, tGammas, selected = gen
    # firstly get the chosen actions at the next timestep
    actions_from_q1 = sess.run(mainQN.predict,feed_dict={mainQN.state:next_states, mainQN.phase:0,\
                                                         mainQN.pred_res:pred_res,mainQN.batch_size:len(states)})
    # Q values for the next timestep from target network, as part of the Double DQN update
    Q2 = sess.run(targetQN.q_output,feed_dict={targetQN.state:next_states, targetQN.phase:0, mainQN.pred_res:pred_res, targetQN.batch_size:len(states)})
    # handles the case when a trajectory is finished
    end_multiplier = 1 - done_flags
    # target Q value using Q values from target, and actions from main
    double_q_value = Q2[range(len(Q2)), actions_from_q1]

    # definition of target Q
    if 'Expo' in env.character or 'Hyper' in env.character: #'TBD' in env.character:
        targetQ = rewards + (tGammas * double_q_value * end_multiplier)            
    else:
        targetQ = rewards + (env.gamma * double_q_value * end_multiplier)

    # get the output q's, actions, and loss
    q_output, actions_taken, abs_error = sess.run([mainQN.q_output,mainQN.predict, mainQN.abs_error], \
                                                  feed_dict={mainQN.state:states,mainQN.targetQ:targetQ, mainQN.actions:actions,
                                               mainQN.phase:False, mainQN.pred_res:pred_res, mainQN.batch_size:len(states)})

    # return the relevant q values and actions
    if env.DEBUG:
        print("actions: {}".format(actions[:3]))
        print("taken_action: {}\n q_outout: {}".format(actions_taken[:3], q_output[:3]))

    phys_q = q_output[range(len(q_output)), actions]
    agent_q = q_output[range(len(q_output)), actions_taken]

    # update the return vals
    error_ret.extend(abs_error)
    all_q_ret.extend(q_output)
    phys_q_ret.extend(phys_q) 
    agent_q_ret.extend(agent_q)
    actions_taken_ret.extend(actions_taken)

    return all_q_ret, phys_q_ret, agent_q_ret, actions_taken_ret, error_ret, selected


def do_eval_pdqn_lstm(sess, env, mainQN, targetQN, testdf, testAll): 
    
    np.set_printoptions(precision=2)
    
    if env.streamNum >= 1:
        for i in range(env.streamNum):
            gen = testAll[i] 
            all_q,phys_q,agent_q,actions_taken,error, selected = do_eval_pdqn_split(sess, env, mainQN, targetQN, gen, i) 

            #print("do_eval_pdqn:idx {}, actions_taken {}, agent_q {}".format(len(idx),len(actions_taken_ret),len(agent_q_ret)))
            testdf.loc[selected.index, 'target_action'] = actions_taken
            testdf.loc[selected.index, 'target_q'] = agent_q
            testdf.loc[selected.index, 'phys_q'] = phys_q
            testdf.loc[selected.index, 'error'] = error
            testdf.loc[selected.index, env.Qfeat] = np.array(all_q)  # save all_q to dataframe      
            #print(actions_taken[:3])
            #print(testdf.loc[selected.index, env.Qfeat[8:26]].head(3))
    else:
        print("No case for the current splitting (see do_eval_pdqn)")
        return
    
    ecr_ret = testdf.groupby(env.pid).head(1).target_q
        
    return testdf, ecr_ret

def do_eval_sarsa(sess, env, mainQN, targetQN, df, traindf):
    all_q_ret = []
    phys_q_ret = []
    actions_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    error_ret = 0
    ecr = []

    gen = process_eval_batch(env, df, df) # use the whole data
    for b in gen: # b: the trajectory for one visit
        states, actions, rewards, next_states, next_actions, done_flags, _ = b
        # firstly get the chosen actions at the next timestep
        actions_from_q1 = sess.run(mainQN.predict,feed_dict={mainQN.state:next_states, mainQN.phase : 0})
        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = sess.run(targetQN.q_output,feed_dict={targetQN.state:next_states, targetQN.phase : 0})
        # handles the case when a trajectory is finished
        end_multiplier = 1 - done_flags
        # target Q value using Q values from target, and actions from main
        next_state_q = Q2[range(len(Q2)), next_actions]
        # definition of target Q
        targetQ = rewards + (env.gamma * next_state_q * end_multiplier)

        # get the output q's, actions, and loss
        q_output, actions_taken, abs_error = sess.run([mainQN.q_output, mainQN.predict, mainQN.abs_error], \
            feed_dict={mainQN.state:states, mainQN.targetQ:targetQ, mainQN.actions:actions, mainQN.phase:False})

        # return the relevant q values and actions
        phys_q = q_output[range(len(q_output)), actions]    
        agent_q = q_output[range(len(q_output)), actions_taken]
        
#       update the return vals
        error = np.mean(abs_error)
        all_q_ret.extend(q_output)
        phys_q_ret.extend(phys_q)
        agent_q_ret.extend(agent_q)
        actions_taken_ret.extend(actions_taken)   
        
    df.loc[:, 'target_action'] = actions_taken_ret
    df.loc[:, 'target_q'] = agent_q_ret
    df.loc[:, 'phys_q'] = phys_q_ret
    df.loc[:, 'error'] = error_ret
    df.loc[:, Qfeat] = np.array(all_q_ret)  # save all_q to dataframe      
    ecr_ret = df.groupby(env.pid).head(1).target_q
    return df, ecr_ret
    

def do_save_results(sess, mainQN, targetQN, df, val_df, test_df, state_features, next_states_feat, gamma, save_dir):
    # get the chosen actions for the train, val, and test set when training is complete.
    _, _, _, agent_q_train, agent_actions_train, _, ecr = do_eval(sess, env, mainQN, targetQN, df)
    #print ("Saving results - length IS ", len(agent_actions_train))
    with open(save_dir + 'dqn_normal_actions_train.p', 'wb') as f:
        pickle.dump(agent_actions_train, f)
    _, _, _, agent_q_test, agent_actions_test, _, ecr = do_eval(sess, env, mainQN, targetQN, test_df)   
    
    # save everything for later - they're used in policy evaluation and when generating plots
    with open(save_dir + 'dqn_normal_actions_train.p', 'wb') as f:
        pickle.dump(agent_actions_train, f)
#     with open(save_dir + 'dqn_normal_actions_val.p', 'wb') as f:
#         pickle.dump(agent_actions_val, f)
    with open(save_dir + 'dqn_normal_actions_test.p', 'wb') as f:
        pickle.dump(agent_actions_test, f)
        
    with open(save_dir + 'dqn_normal_q_train.p', 'wb') as f:
        pickle.dump(agent_q_train, f)
#     with open(save_dir + 'dqn_normal_q_val.p', 'wb') as f:
#         pickle.dump(agent_q_val, f)
    with open(save_dir + 'dqn_normal_q_test.p', 'wb') as f:
        pickle.dump(agent_q_test, f)
        
    with open(save_dir + 'ecr_test.p', 'wb') as f:
        pickle.dump(ecr, f)    
    return


def do_save_results_sarsa(sess, env, mainQN, targetQN, df, val_df, test_df, save_dir):
    # get the chosen actions for the train, val, and test set when training is complete.
    all_q_ret, phys_q_train, phys_actions_train, _, ecr =  do_eval_sarsa(sess, env, mainQN, targetQN, df)        
    all_q_ret, phys_q_test, phys_actions_test, _, ecr = do_eval_sarsa(sess, env, mainQN, targetQN, test_df)
    
    # save everything for later - they're used in policy evaluation and when generating plots
    with open(save_dir + 'phys_actions_train.p', 'wb') as f:
        pickle.dump(phys_actions_train, f)
    with open(save_dir + 'phys_actions_test.p', 'wb') as f:
        pickle.dump(phys_actions_test, f)
        
    with open(save_dir + 'phys_q_train.p', 'wb') as f:
        pickle.dump(phys_q_train, f)
    with open(save_dir + 'phys_q_test.p', 'wb') as f:
        pickle.dump(phys_q_test, f)
    with open(save_dir + 'ecr_test.p', 'wb') as f:
        pickle.dump(ecr, f)
    return

def check_convergence(df, agent_actions):
    df["agent_actions"] = agent_actions
    Diff_policy = len(df[df.agent_actions != df.agent_actions_old])
    if Diff_policy > 0:
        print("Policy is not converged {}/{}".format(Diff_policy, len(df)))
    elif Diff_policy == 0:
        print("Policy is converged!!")
    df['agent_actions_old'] = df.agent_actions
    return df

#------------------
# Preprocessing

def convert_action(df, col):
    df.loc[df[df[col] == 0].index, col] = 'N'  #0
    df.loc[df[df[col] == 1].index, col] = 'V'  #1
    df.loc[df[df[col] == 2].index, col] = 'A'  #10
    df.loc[df[df[col] == 3].index, col] = 'AV' #11
    df.loc[df[df[col] == 4].index, col] = 'O'  #100
    df.loc[df[df[col] == 5].index, col] = 'OV' #101
    df.loc[df[df[col] == 6].index, col] = 'OA' #110
    df.loc[df[df[col] == 7].index, col] = 'OAV'#111
    return df
    
def action_dist(df, feat):
    for a in action8:
        print("{}\t{}\t({:.2f})".format(a, len(df[df[feat] == a]), len(df[df[feat] == a])/len(df)))

def process_train_batch(df, size, per_flag, state_features, next_states_feat):
    if per_flag:
        # uses prioritised exp replay
        a = df.sample(n=size, weights=df['prob'])
    else:
        a = df.sample(n=size)

    actions = a.loc[:, 'Action'].tolist()
    rewards = a.loc[:, 'reward'].tolist()
    states = a.loc[:, state_features].values.tolist()
    
    # next_actions = a.groupby('VisitIdentifier').Action.shift(-1).fillna(0).tolist()
    next_states = a.loc[:, next_states_feat].values.tolist() #a.groupby('VisitIdentifier')[state_features].shift(-1).fillna(0).values.tolist()
    done_flags = a.done.tolist()
    
    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)


def process_train_batch_sarsa(df, size, per_flag, state_features, next_states_feat):
    if per_flag:   # uses prioritised exp replay
        a = df.sample(n=size, weights=df['prob'])
    else:
        a = df.sample(n=size)

    actions = a.loc[:, 'Action'].tolist()
    rewards = a.loc[:, 'reward'].tolist()
    states = a.loc[:, state_features].values.tolist()
    next_actions = a.loc[:, 'next_action'].tolist() #np.array(a.groupby('VisitIdentifier').Action.shift(-1).fillna(0), dtype=np.int).tolist()
    next_states =  a.loc[:, next_states_feat].values.tolist() #a.groupby('VisitIdentifier')[state_features].shift(-1).fillna(0).values.tolist()
    done_flags = a.done.tolist()
    
    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(next_actions), np.squeeze(done_flags), a)    
# ---------------
# Analysis
def compareSimilarity(df):
    pid = 'VisitIdentifier' 
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    col = 'Sim_Action'
    df[col] = 0
    df.loc[df[df.Phys_Action == df.Target_Action].index, col] = 1
    df.loc[df[((df.Phys_Action=='V')&(df.Target_Action=='AV'))|((df.Phys_Action=='V')&(df.Target_Action=='OV'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='AV'))|((df.Phys_Action=='A')&(df.Target_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='O')&(df.Target_Action=='OV'))|((df.Phys_Action=='O')&(df.Target_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='OAV'))|((df.Phys_Action=='O')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='V')&(df.Target_Action=='OAV'))].index,col] = 1/3
    df.loc[df[((df.Target_Action=='V')&(df.Phys_Action=='AV'))|((df.Target_Action=='V')&(df.Phys_Action=='OV'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='AV'))|((df.Target_Action=='A')&(df.Phys_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='O')&(df.Phys_Action=='OV'))|((df.Target_Action=='O')&(df.Phys_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='OAV'))|((df.Target_Action=='O')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='V')&(df.Phys_Action=='OAV'))].index,col] = 1/3
    df.loc[df[((df.Target_Action=='OA')&(df.Phys_Action=='OAV'))|((df.Target_Action=='OV')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='AV')&(df.Phys_Action=='OAV'))].index,col] = 2/3
    df.loc[df[((df.Phys_Action=='OA')&(df.Target_Action=='OAV'))|((df.Phys_Action=='OV')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='AV')&(df.Target_Action=='OAV'))].index,col] = 2/3
    
    # Result Analysis 
    rdf = (df.groupby(pid)[col].sum() / df.groupby(pid).size()).reset_index(name='Sim_ratio').sort_values(['Sim_ratio'], ascending=False)
    rdf['Shock'] = 0
    posvids = df[df.Shock == 1].VisitIdentifier.unique()
    rdf.loc[rdf[pid].isin(posvids), 'Shock']  = 1
    
    return df, rdf

def compareDefference(df):
    pid = 'VisitIdentifier' 
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df['Diff_Action'] = 0
    df.loc[df[df.Phys_Action != df.Target_Action].index, 'Diff_Action'] = 1
    df.loc[df[((df.Phys_Action=='V')&(df.Target_Action=='AV'))|((df.Phys_Action=='V')&(df.Target_Action=='OV'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='AV'))|((df.Phys_Action=='A')&(df.Target_Action=='OA'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Phys_Action=='O')&(df.Target_Action=='OV'))|((df.Phys_Action=='O')&(df.Target_Action=='OA'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='OAV'))|((df.Phys_Action=='O')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='V')&(df.Target_Action=='OAV'))].index,'Diff_Action'] = 0.6
    df.loc[df[((df.Target_Action=='V')&(df.Phys_Action=='AV'))|((df.Target_Action=='V')&(df.Phys_Action=='OV'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='AV'))|((df.Target_Action=='A')&(df.Phys_Action=='OA'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Target_Action=='O')&(df.Phys_Action=='OV'))|((df.Target_Action=='O')&(df.Phys_Action=='OA'))].index,'Diff_Action'] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='OAV'))|((df.Target_Action=='O')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='V')&(df.Phys_Action=='OAV'))].index,'Diff_Action'] = 0.6
    df.loc[df[((df.Target_Action=='OA')&(df.Phys_Action=='OAV'))|((df.Target_Action=='OV')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='AV')&(df.Phys_Action=='OAV'))].index,col] = 0.3  
    df.loc[df[((df.Phys_Action=='OA')&(df.Target_Action=='OAV'))|((df.Phys_Action=='OV')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='AV')&(df.Target_Action=='OAV'))].index,col] = 0.3

    # Result Analysis 
    rdf = (df.groupby(pid).Diff_Action.sum() / df.groupby(pid).size()).reset_index(name='Diff_ratio').sort_values(['Diff_ratio'], ascending=False)
    rdf['Shock'] = 0
    posvids = df[df.Shock == 1].VisitIdentifier.unique()
    rdf.loc[rdf[pid].isin(posvids), 'Shock']  = 1

    return df, rdf

def getPolicySimilarity(df, div):
    df.rename(columns = {'Agent_Action': 'Target_Action'}, inplace=True)    
    _, resdf = compareSimilarity(df)
    resdf, shockSimRate, nonShockSimRate, avgSimRate = getPolicySimRate(resdf, div)
    return resdf, shockSimRate, nonShockSimRate, avgSimRate

def getPolicySimRate(rdf, div): # div: how many 
    col = 'Sim_ratio'
    res = pd.DataFrame(columns=['SimRate', 'pos', 'neg', 'shockRate'])
    avgSimRate = rdf[col].mean()
    shockSimRate = rdf[rdf.Shock==1][col].mean()
    nonShockSimRate = rdf[rdf.Shock==0][col].mean()
    print("Similarity Rate: avg({:.4f}), shock({:.4f}), non-shock({:.4f})".format(avgSimRate, shockSimRate, nonShockSimRate))
    #print("DiffRate PosNum\tNegNum\tShockRate")   
    for i in range(div):
        s0 = len(rdf[(rdf[col] < (i+1)/div) & (rdf[col] >= i/div) & (rdf.Shock == 0)]) 
        s1 = len(rdf[(rdf[col] < (i+1)/div) & (rdf[col] >= i/div) & (rdf.Shock == 1)])
        if s0+s1 > 0:
            res.loc[len(res),:] = [(i)/div, s1, s0, s1/(s0+s1)]
            
    s0 = len(rdf[(rdf[col] == 1) & (rdf.Shock == 0)]) 
    s1 = len(rdf[(rdf[col] == 1) & (rdf.Shock == 1)])
    if s0+s1 > 0:
        res.loc[len(res),:] = [1, s1, s0, s1/(s0+s1)]
    return res, shockSimRate, nonShockSimRate, avgSimRate

def getPolicyDiffRate(rdf, div):
    resDiff = pd.DataFrame(columns=['diffRate', 'pos', 'neg', 'shockRate'])
    avgDiffRate = rdf.Diff_ratio.mean()
    shockDiffRate = rdf[rdf.Shock==1].Diff_ratio.mean()
    nonShockDiffRate = rdf[rdf.Shock==0].Diff_ratio.mean()
    print("Difference Rate: avg({:.4f}), shock({:.4f}), non-shock({:.4f})".format(avgDiffRate, shockDiffRate, nonShockDiffRate))

    for i in range(div):
        s0 = len(rdf[(rdf.Diff_ratio < (i+1)/div) & (rdf.Diff_ratio >= i/div) & (rdf.Shock == 0)]) 
        s1 = len(rdf[(rdf.Diff_ratio < (i+1)/div) & (rdf.Diff_ratio >= i/div) & (rdf.Shock == 1)])
        if s0+s1 > 0:
            #print("{:0.2f}\t {:} \t{:} \t{:0.2f}".format((i)/div, s1, s0, s1/(s0+s1)))
            resDiff.loc[len(resDiff),:] = [(i)/div, s1, s0, s1/(s0+s1)]

    s0 = len(rdf[(rdf.Diff_ratio == 1) & (rdf.Shock == 0)]) 
    s1 = len(rdf[(rdf.Diff_ratio == 1) & (rdf.Shock == 1)])
    if s0+s1 > 0:
        resDiff.loc[len(resDiff),:] = [1, s1, s0, s1/(s0+s1)]
    return resDiff, avgDiffRate, shockDiffRate, nonShockDiffRate
        
def showTrajectoryLength(df):
    groupSize = df.groupby('VisitIdentifier').size()
    print("Length of Trajectory: mean({:.2}), max({}), min({})".format(groupSize.mean(), groupSize.max(), groupSize.min()))
        
def rl_analysis(df, target_actions, all_q_ret, target_q, availableActions):
    pid = 'VisitIdentifier' 
    df.loc[:, 'target_action'] = target_actions
    df.loc[:, 'target_q'] = target_q

    # save all_q to dataframe
    for i in range(np.size(all_q_ret, 1)):
        df['Q'+str(i)] = np.array(all_q_ret)[:,i]

    df['Target_Action'] = np.nan
    df['Phys_Action'] = np.nan
    idx = 0
    for a in availableActions: 
        df.loc[df[df.target_action == idx].index, 'Target_Action'] = a
        df.loc[df[df.Action == idx].index, 'Phys_Action'] = a
        idx += 1    

    action_num = len(df.Action.unique())
    df.loc[:, 'rand_action'] = [random.randint(0, action_num-1) for _ in range(len(df))]
    
    df['Diff_Action'] = 0
    df.loc[df[df.Action != df.target_action].index, 'Diff_Action'] = 1

    # Result Analysis 
    rdf = (df.groupby(pid).Diff_Action.sum() / df.groupby(pid).size()).reset_index(name='Diff_ratio').sort_values(['Diff_ratio'], ascending=False)
    rdf['Shock'] = 0
    posvids = df[df.Shock == 1].VisitIdentifier.unique()
    rdf.loc[rdf[pid].isin(posvids), 'Shock']  = 1
    # Diff_ratio by range: shock probability (but underlying distribution?)
    # [0~0.1), [0.1, 0.2), ... [0.9, 1], [1] 
    
    return df, rdf

def rl_analysis_pdqn(env, df): 
    df.loc[:,'Target_Action'] = np.nan
    df.loc[:,'Phys_Action'] = np.nan
    idx = 0
    for a in env.actions: 
        df.loc[df[df.target_action == idx].index, 'Target_Action'] = a
        df.loc[df[df.Action == idx].index, 'Phys_Action'] = a
        idx += 1    

    action_num = len(df.Action.unique())
    df.loc[:, 'rand_action'] = [random.randint(0, action_num-1) for _ in range(len(df))]
    
    df.loc[:,'Diff_Action'] = 0
    df.loc[df[df.Action != df.target_action].index, 'Diff_Action'] = 1
    #compute the septic shock ratio according to the difference

    # Result Analysis 
    rdf = (df.groupby(env.pid).Diff_Action.sum() / df.groupby(env.pid).size()).reset_index(name='Diff_ratio').sort_values(['Diff_ratio'], ascending=False)
    rdf[env.label] = 0
    posvids = df[df[env.label] == 1][env.pid].unique()
    rdf.loc[rdf[env.pid].isin(posvids), 'Shock']  = 1
    # Diff_ratio by range: shock probability (but underlying distribution?)
    # [0~0.1), [0.1, 0.2), ... [0.9, 1], [1] 

    return df, rdf

def rewardNonShock(df, negvids, reward):
    ndf = df[df[pid].isin(negvids)]
    df.loc[ndf.groupby(pid).tail(1).index, 'reward'] = reward # non-shock: positive reward
    return df

def extractResults(feat, phy, phy_ir, dqn, dqn_ir):
    rdf = pd.DataFrame(columns = ['timestep', 'Physician', 'Physician_IR', 'DQN', 'DQN_IR'])
    rdf.loc[:, 'timestep'] = phy.timestep
    rdf.loc[:, 'Physician'] = phy[feat]
    rdf.loc[:, 'Physician_IR'] = phy_ir[feat]
    rdf.loc[:, 'DQN'] = dqn[feat]
    rdf.loc[:, 'DQN_IR'] = dqn_ir[feat]
    return rdf

def mergeResults(path, keyword):
    phy = pd.read_csv(path+"log_"+keyword+"_phy_SR.csv", header=0)
    phy_ir = pd.read_csv(path+"log_"+keyword+"_phy_IR.csv", header=0)
    dqn = pd.read_csv(path+"log_"+keyword+"_dqn_SR.csv", header=0)
    dqn_ir = pd.read_csv(path+"log_"+keyword+"_dqn_IR.csv", header=0)

    avg_q = extractResults('avg_Q', phy, phy_ir, dqn, dqn_ir)
    mae = extractResults('MAE', phy, phy_ir, dqn, dqn_ir)
    avg_loss = extractResults('avg_loss', phy, phy_ir, dqn, dqn_ir)

    avg_q.to_csv(path+keyword+"_avg_Q.csv", index=False)
    mae.to_csv(path+keyword+"_mae.csv", index=False)
    avg_loss.to_csv(path+keyword+"_avg_loss.csv", index=False)

# ---------------------------------------------------------------------------- 
# Prediction

def prep_predData(file, feat, taus, pid):
    df = pd.read_csv(file, header=0)
    df = df[['VisitIdentifier', 'MinutesFromArrival']+feat]
    df = df.drop(df.loc[df.MinutesFromArrival < 0].index)  # Drop negative MinutesFromArrival events
    # get the idx to make the prediction data (with 1-hour time window)
    df['agg_idx'] = np.abs(df.MinutesFromArrival // 60) 
    df['pred_idx'] = 0
    df.loc[(df.shift(-1).agg_idx - df.agg_idx) != 0, 'pred_idx'] = 1 
    
    df = tl.setmi(df, feat) # set missing indicators
    df = tl.make_belief(df, pid, feat, taus, mode='.75') # impute 
    return df

# TBM imputation + Add MI
def init_predict(pid):
    filepath = '../../rl/data/preproc/'    
    taus_org = pd.read_csv('data/pdqn_final_LSTM/timedf0h.csv', header = 0)
    taus = taus_org.loc[1, :][1:]
    feat = taus_org.columns[1:].tolist()
    totfeat = [] # for prediction    
    for f in feat:
        totfeat.append(f)
        totfeat.append(f+'_mi')
    pred_train_df = prep_predData(filepath+"3_3_beforeShock_Prediction_Train_0123.csv", feat, taus, pid)
    pred_test_df = prep_predData(filepath+"3_3_beforeShock_Prediction_Test_0123.csv", feat, taus, pid)

    return taus, feat, totfeat, pred_train_df, pred_test_df 

# make data with prediction index (1-hour aggregation) 
def makeXY_idx(pred_df, feat, pid, label, MRL): 
    X = []
    Y = []
    posvids = pred_df[pred_df[label] == 1][pid].unique()
    eids = pred_df[pid].unique()
    for eid in eids:
        edf = pred_df[pred_df[pid] == eid]
        tmp = np.array(edf[feat])       
        indexes = edf[edf.pred_idx ==1].index
        if eid in posvids:
            Y += [1]*len(indexes)
        else:
            Y += [0]*len(indexes)

        for i in indexes:
            X.append(pad_sequences([tmp[:i+1]], maxlen = MRL, dtype='float'))

    return X, Y

# get prediction values (Currently used)
from keras import backend as K
def get_prediction(df, test_df, pid, label, MRL):
    MRL = 20
    mi_mode = True
    fill_mode = '.75'    
    # initialize the prediction data
    taus, feat, totfeat, pred_train_df, pred_test_df  = init_predict(pid) # TBM imputation + add MI
    print("RL data: train({}), test({}) / Pred data idx: train({}), test({})".format(len(df),len(test_df),pred_train_df.pred_idx.sum(), pred_test_df.pred_idx.sum()))
    
    # make X, Y data for prediction (RL data and XY data have same indexes)
    test_Xpad, test_Ypad = makeXY_idx(pred_test_df, totfeat, pid, label, MRL)
    train_Xpad, train_Ypad = makeXY_idx(pred_train_df, totfeat, pid, label, MRL)
    
    df['pred_val'] = np.nan
    test_df['pred_val'] = np.nan
    train_pred_val = []
    test_pred_val = []
    with tf.Session() as sess_pred:
        K.set_session(sess_pred)
        pred_model = tl.load_model('data/pdqn_final_LSTM/models/model_final_.75True')
        pred_model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics = ['binary_accuracy'])
        #loss, acc = pred_model.test_on_batch(np.expand_dims(train_Xpad[0][0], axis=0), np.array([Y[0]]))
        for j in df.index:
            train_pred_val.append(pred_model.predict_on_batch(np.expand_dims(train_Xpad[j][0], axis=0))[0][0])
            if j % 10000 == 0:
                print(j)
        df['pred_val'] = train_pred_val
        for j in test_df.index:
            test_pred_val.append(pred_model.predict_on_batch(np.expand_dims(test_Xpad[j][0], axis=0))[0][0])
            if j % 10000 == 0:
                print(j)
        test_df['pred_val'] = test_pred_val

    df.to_csv("../data/preproc/10_BS_pred_train.csv", index=False)
    test_df.to_csv("../data/preproc/10_BS_pred_test.csv", index=False)
    return df, test_df

# B. Event-level sequence data generation
# predict the next label: shift the labels with 1 timestep backward
def makeXY_event_label(df, feat, pid, label, MRL): 
    X = []
    Y = []
    posvids = df[df[label] == 1][pid].unique()
    eids = df[pid].unique()
    for eid in eids:
        edf = df[df[pid] == eid]
        tmp = np.array(edf[feat])
        
        for i in range(len(tmp)):
            X.append(pad_sequences([tmp[:i+1]], maxlen = MRL, dtype='float')) 
            
        if eid in posvids: # generate event-level Y labels based on the ground truth
            Y += [1]*len(edf)
        else:
            Y += [0]*len(edf)
    print("df:{} - Xpad:{}, Ypad{}".format(len(df), np.shape(X), np.shape(Y)))
    return np.array(X), np.array(Y)

# B. Event-level sequence data generation
# predict the next label: shift the labels with 1 timestep backward
def makeXY_event_label2(df, feat, pid, label, MRL): 
    X = []
    Y = []
    posvids = df[df[label] == 1][pid].unique()
    eids = df[pid].unique()
    for eid in eids:
        edf = df[df[pid] == eid]
        tmp = np.array(edf[feat])

        for i in range(len(edf)):
            X.append(np.array(tmp[:i+1]))#, maxlen = MRL, dtype='float')) 
            
        if eid in posvids: # generate event-level Y labels based on the ground truth
            Y += [1]*len(edf)
        else:
            Y += [0]*len(edf)
   
    X = pad_sequences(X, maxlen = MRL, dtype='float')
  
    return X, Y

# USED for LSTM data generation (0819.2019)
def getLSTMdata(df, feat, MRL, filename):
    startTime = time.time()
    X = makeX_event_given_batch(df, df, feat, pid, MRL) # makeX_event_random_batch
    print("X: {} ({:.1f} min)".format(np.shape(X), (time.time() - startTime)/60))
    with open(filename+'_t'+str(MRL)+'.pk', 'wb') as fp:
        pickle.dump(X, fp)
    return X


def makeX_event_given_batch_event(df, a, feat, pid, MRL): 
    X = []
    for i in a.index.tolist():
        edf = df[df[pid] == a.loc[i, pid]]
        X.append(np.array(edf.loc[:i, feat]))

    X = pad_sequences(X, maxlen = MRL, dtype='float')
    #print("X:", np.shape(X))
    return X


# for all the selected data
def makeX_event_given_batch(df, a, feat, pid, MRL): 
    X = []
    eids = a[pid].unique().tolist()
    idx = a.index.tolist()

    for i in range(len(eids)):
        edf = df[df[pid] == eids[i]]
        tmp = np.array(edf[feat])
        for j in range(len(edf)):
            X.append(np.array(tmp[:j+1]))            
    X = pad_sequences(X, maxlen = MRL, dtype='float')
    return X

def getData(path, trainfile, testfile, state_features, available_actions, pid, timeFeat):
    df, nextStatesFeat = ld.initEHRdata2(path+trainfile+".csv", state_features, available_actions, pid, timeFeat)
    test_df, _ = ld.initEHRdata2(path+testfile+".csv", state_features, available_actions, pid, timeFeat)

    return df, test_df, nextStatesFeat


def imputeTBM(df, pid, fill_mode, outfile):
    staticFeat = ['Gender', 'Age', 'Race'] 
    taus_org = pd.read_csv('data/pdqn_final_LSTM/timedf0h.csv', header = 0)
    taus = taus_org.loc[1, :][1:]
    feat = taus_org.columns[1:].tolist()#[:14]
    df = tl.make_belief_mean(df, pid, feat, taus, mode='.75')    
    df.to_csv(outfile, index=False)
    
    totfeat = [] # for prediction    
    for f in feat:
        totfeat.append(f)
        totfeat.append(f+'_mi')
    return df, feat, totfeat

def imputeTBM_GAR(df, pid, numFeat, staticFeat, fill_mode, mi_mode, taufile, outfile, outWindow):      
    print("load TBM imputed data with GAR")
    #numFeat = ['HeartRate', 'Temperature', 'SystolicBP', 'MAP', 'Lactate', 'WBC', 'Platelet', 'Creatinine',
    #   'RespiratoryRate', 'FIO2', 'PulseOx', 'BiliRubin', 'BUN',  'Bands']#The order should be fixed for both prediction and RL
    #staticFeat = ['Gender', 'Age', 'Race'] 
    taus_org = pd.read_csv(taufile, header = 0)
    taus = taus_org.loc[1, :][1:]
    tauFeat = taus_org.columns[1:].tolist()#[:14]
    if outWindow == 'zero':
        df = tl.make_belief(df, pid, tauFeat, taus, mode='.75')  #FOor both standardization and normalization case, zero-impute should be done out of the reliable time windows. (Here the meaning is that we assume when we cannot rely on the previous features, we just exclude thouse missing features and let LSTM infer it only based on other features. It'll find the relationalship.      
    elif outWindow == 'mean':
        df = tl.make_belief_mean(df, pid, tauFeat, taus, mode='.75')    
    
    df.to_csv(outfile, index=False)
   
    if mi_mode == True:
        df = tl.setmi(df, numFeat) # set missing indicators
        totfeat = [] # for prediction    
        for f in numFeat:
            totfeat.append(f)
            totfeat.append(f+'_mi')
        totfeat += staticFeat
        feat = numFeat + staticFeat
    else:
        feat = numFeat + staticFeat
        totfeat = feat
    return df, feat, totfeat

def get_prediction_raw(df, pid, label, totfeat, pred_model, MRL, outfile):  
    # make X, Y data for prediction (RL data and XY data have same indexes)
    print("make X, Y with event level labels")
    Xpad, Ypad = makeXY_event_label(df, totfeat, pid, label, MRL)
    df['visitShock'] = Ypad
    df['pred_val'] = np.nan
    pred_val = []
    pred_Y = []
    print("Prediction start...")
    with tf.Session() as sess_pred:
        K.set_session(sess_pred)
        #pred_model = tl.load_model('data/pdqn_GAR_LSTM/models/model_final_.75True')
        pred_model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics = ['binary_accuracy'])
        #loss, acc = pred_model.test_on_batch(np.expand_dims(train_Xpad[0][0], axis=0), np.array([Y[0]]))
        for j in df.index:
            pred_val.append(pred_model.predict_on_batch(np.expand_dims(Xpad[j][0], axis=0))[0][0])
            if j % 5000 == 0 and j > 0:
                print(j)
               
        df['pred_val'] = pred_val
        print("{}: pos - pred {} / ground {} with 0.5".format(j, len(df[df.pred_val>=0.5]), np.sum(Ypad)))

        if outfile != '':
            df.to_csv(outfile, index=False)
    return df


def prediction(X, Y, batch_indexes, model): #, bf_mode=True, mi_mode=mi_mode, totfeat=totfeat, fill_mode=fill_mode):
    predY = []
    trueY = []       
    pred = []
    avg_loss = 0
    avg_acc = 0
    trueY = np.array([Y[i] for i in batch_indexes])
    for j in batch_indexes:
        loss, acc = model.test_on_batch(np.expand_dims(X[j][0], axis=0), np.array([Y[j]]))
        pred_val = model.predict_on_batch(np.expand_dims(X[j][0], axis=0))[0][0]
        predY.append(int(round(pred_val))) # binary prediction
        pred.append(pred_val) # real value of prediction
        avg_loss += loss
        avg_acc += acc
    model.reset_states()
    avg_loss /= len(batch_indexes)
    avg_acc /= len(batch_indexes)
    #K.clear_session()
    return pred, predY, trueY, avg_loss, avg_acc

