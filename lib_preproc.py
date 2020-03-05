# Library for preprocessing
# author: Yeo Jin Kim
# date: 8/28/2018

import pandas as pd
import numpy as np
from scipy import stats
import os
import pickle

from keras.preprocessing.sequence import pad_sequences


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


def getLSTMdata(df, feat, MRL, filename):
    pid = 'VisitIdentifier'
    X = makeX_event_given_batch(df, df, feat, pid, MRL) # makeX_event_random_batch
    if filename != '':
        with open(filename+'_t'+str(MRL)+'.pk', 'wb') as fp:
            pickle.dump(X, fp)
    return X

# Inverse the mean and std from standardized data
def getMeanStd(df, feat, feat_org):
    tmpIdx = df[pd.notnull(df[feat_org])].index[:11].tolist()
    #print(feat, tmpIdx)
    for i in range(10):
        if df.loc[tmpIdx[0], feat] != df.loc[tmpIdx[i+1], feat]:
            idx = [tmpIdx[0], tmpIdx[i+1]]
            break
    f1_org = df.loc[idx[0], feat_org]
    f2_org = df.loc[idx[1], feat_org]
    f1 = df.loc[idx[0], feat]
    f2 = df.loc[idx[1], feat]
    f_std = (f1_org-f2_org)/(f1-f2)
    f_mean = f1_org - f1*f_std
    return f_mean, f_std


def splitData(df):
    pid = 'VisitIdentifier' 
    posvids, negvids, totvids = statLabels(df, 'Shock', 'data', pid)
    pos_size = int(len(posvids)/5)
    neg_size = int(len(negvids)/5)
    pdf = df[df.VisitIdentifier.isin(posvids)]
    ndf = df[df.VisitIdentifier.isin(negvids)]
    test_df = pdf[pdf[pid].isin(posvids[:pos_size])].append(ndf[ndf[pid].isin(negvids[:neg_size])])
    test_df = test_df.reset_index(drop=True)
    val_df = test_df
    val_df['agent_actions_old'] = np.nan
    df = pdf[pdf[pid].isin(posvids[pos_size:])].append(ndf[ndf[pid].isin(negvids[neg_size:])])
    df = df.reset_index(drop=True)
    print("split the given data to train(80%) and test(20%) data")
    return df, val_df, test_df   
                               
# 1. merge data + shock labels
def mergeDataAllLabels(datafile, labelfile, outfile,pid):
    print("merging labels")
    rdf = pd.read_csv(datafile, header=0, usecols=lf.feats_org)
    ldf = pd.read_csv(labelfile, header=0, usecols=['VisitIdentifier', 'MinutesFromArrival', 'InfectionFlag', 'InflammationFlag', 'OrganFailure','ShockFlag'])
    mdf = pd.merge(rdf, ldf, on=['VisitIdentifier', 'MinutesFromArrival'])
    mdf = mdf.sort_values(['VisitIdentifier', 'MinutesFromArrival'])
    mdf.to_csv(outfile, index=False)
    posvids, negvids, totvids = statLabels(mdf, 'ShockFlag', 'NewVisitEvent for RL', pid)
    return mdf, posvids, negvids, totvids
    
def mergeDataLabel(datafile, labelfile, outfile,pid):
    print("merging labels")
    rdf = pd.read_csv(datafile, header=0, usecols=lf.feats_org)
    ldf = pd.read_csv(labelfile, header=0, usecols=['VisitEventID', 'ShockC1', 'ShockC2', 'ShockC3', 'ShockC4','ShockFlag'])
    mdf = pd.merge(rdf, ldf, on='VisitEventID')
    mdf = mdf.sort_values(['VisitIdentifier', 'MinutesFromArrival'])
    mdf.to_csv(outfile, index=False)
    posvids, negvids, totvids = statLabels(mdf, 'ShockFlag', 'NewVisitEvent for RL', pid)
    return mdf, posvids, negvids, totvids

def mergeStageLabel(df, stagefile, outfile, pid):
    print("merging stage labels")
    ldf = pd.read_csv(labelfile,header=0, usecols=['VisitEventID', 'InfectionFlag', 'InflammationFlag', 'OrganFailure'])
    mdf = pd.merge(rdf, ldf, on='VisitEventID')
    mdf = mdf.sort_values(['VisitIdentifier', 'MinutesFromArrival'])
    mdf.to_csv(outfile, index=False)
    return mdf

   
# Intersect both pos and both neg in ICD9 & Rules
def intersectLabels(df, vdf, posvid, negvid, totvids):
    print("* intersection of icd9 and rules")
    icd9Tvid = vdf.VisitIdentifier.tolist()
    icd9Fvid = [vid for vid in totvids if vid not in icd9Tvid]
    #print("icd9pos_vids:", len(icd9pos_vids))
    df['ShockICD9'] = 0
    df.loc[df[df.VisitIdentifier.isin(icd9Tvid)].index,'ShockICD9'] = 1
    print("ShockICD9==1:", len(df[df.ShockICD9==1].VisitIdentifier.unique()))
    df['Shock'] = np.nan
    df.loc[df[(df.ShockICD9 == 1) & (df.ShockFlag == 1)].index, 'Shock'] = 1
    df.loc[df[(df.ShockICD9 == 0) & (df.ShockFlag == 0)].index, 'Shock'] = 0

    double_posvid = df[df.Shock ==1].VisitIdentifier.unique().tolist()    
    double_negvid = [x for x in negvids if x in icd9Fvid]
    vid_iTrF = [x for x in negvids if x in icd9Tvid]
    vid_iFrT = [x for x in posvids if x in icd9Fvid]    
    
    #double_negvid = df[df.Shock ==0].VisitIdentifier.unique().tolist()
#     vid_iTrF = len(df[(df.ShockICD9 ==1)&(df.ShockFlag==0)].VisitIdentifier.unique())
#     vid_iFrT = len(df[(df.ShockICD9 ==0)&(df.ShockFlag==1)].VisitIdentifier.unique())
#     icd9Tvid = len(df[df.ShockICD9 ==1].VisitIdentifier.unique())
#     icd9Fvid = len(df[df.ShockICD9 ==0].VisitIdentifier.unique())
    statcols = ['Rule_T', 'Rule_F', 'total']
    statrows = ['ICD9_T', 'ICD9_F', 'total']
    data = [[len(double_posvid), len(vid_iTrF), len(icd9Tvid)],
            [len(vid_iFrT),len(double_negvid), len(icd9Fvid)],
            [len(posvid), len(negvid), len(totvids)]]
    print(pd.DataFrame(data, statrows, statcols))

#     icd9Tvid = vf.VisitIdentifier.tolist() #vf[vf['SepticShockICD9']==1].VisitIdentifier.tolist()
#     icd9Fvid = [vid for vid in totvids if vid not in icd9Tvid]
#     double_posvid = [x for x in posvid if x in icd9Tvid]
#     double_negvid = [x for x in negvid if x in icd9Fvid]
#     vid_iTrF = [x for x in negvid if x in icd9Tvid]
#     vid_iFrT = [x for x in posvid if x in icd9Fvid]
#     statcols = ['Rule_T', 'Rule_F', 'total']
#     statrows = ['ICD9_T', 'ICD9_F', 'total']
#     data = [[len(double_posvid), len(vid_iTrF), len(icd9Tvid)],
#             [len(vid_iFrT),len(double_negvid), len(icd9Fvid)],
#             [len(posvid), len(negvid), len(posvid)+len(negvid)]]
#     print(pd.DataFrame(data, statrows, statcols))

    # 1-3. visit sampling 
    sdf = df[df.VisitIdentifier.isin(double_posvids+double_negvids)]
    print("total selected vids: ", len(sdf.VisitIdentifier.unique()))
    sdf.to_csv("data/3_rldata_intersect_selected_0705.csv", index=False)
    return sdf, double_posvid, double_negvid    
    
def t_test(pid_visit, pid_event, sample, population, visit, event, len_equal = False, debug=False):

    selected = visit.GenderDescription == 'Unknown'   # why unknown is Female?
    visit.loc[selected, 'GenderDescription'] = 'Female'

    if debug == True:
        print("Number of sample")
        print(len(sample))

    selected = visit[pid_visit].isin(sample)
    nonshock_sample_visit = visit[selected]
    selected = visit[pid_visit].isin(population)
    nonshock_population_visit = visit[selected]

    t1 = stats.ttest_ind(nonshock_sample_visit.AgeCategory, nonshock_population_visit.AgeCategory, equal_var= len_equal)[1]
    t2 = stats.ttest_ind(nonshock_sample_visit.LOSDays, nonshock_population_visit.LOSDays, equal_var= len_equal)[1]

    if debug == True:
        print(nonshock_population_visit.groupby(['GenderDescription']).size())
        print(nonshock_sample_visit.groupby(['GenderDescription']).size())
        print(nonshock_population_visit.groupby(['EthnicGroupDescription']).size())
        print(nonshock_sample_visit.groupby(['EthnicGroupDescription']).size())


    t3 = stats.chisquare(nonshock_sample_visit.groupby(['GenderDescription']).size().apply(lambda x: 100 * x / float(nonshock_sample_visit.shape[0])),
                         nonshock_population_visit.groupby(['GenderDescription']).size().apply(lambda x: 100 * x / float(nonshock_population_visit.shape[0])))[1]

    t4 = stats.chisquare(nonshock_sample_visit.groupby(['EthnicGroupDescription']).size().apply(lambda x: 100 * x / float(nonshock_sample_visit.shape[0])),
                         nonshock_population_visit.groupby(['EthnicGroupDescription']).size().apply(lambda x: 100 * x / float(nonshock_population_visit.shape[0])))[1]

    selected = event[pid_event].isin(sample)
    nonshock_sample_event = event[selected]
    selected = event[pid_event].isin(population)
    nonshock_population_event = event[selected]


    t5 = stats.ttest_ind(nonshock_sample_event.groupby(['VisitIdentifier']).size(),
                         nonshock_population_event.groupby(['VisitIdentifier']).size())[1]


    if debug == True:
        print("T test result for age, LOSDays, gender and Ethnic Group:")
        print(t1)
        print(t2)
        print(t3)
        print(t4)
        print(t5)

    if (t1>0.5) and (t2>0.5) and (t3>0.5) and (t4>0.5):
        return False
    else:
        return True

# distribution of gender, age, LOS, race, & ethnic
def distGALRE(df):
    #target_race = ['White', 'Black or African American', 'Asian', 'American Indian or Alaska Native']
    #['Undefined','Unavailable', 'Decline', 'Unknown','Pacific Islander','Multi Racial','Native Hawaiian/other Pacific Islander']
    print("Gender          M({:.0f}) F({:.0f}) U({:.0f})".format(len(df[df.GenderDescription=='Male']),
                                                     len(df[df.GenderDescription=='Female']),
                                                     len(df[df.GenderDescription=='Unknown'])))
    print("Age             {:.2f} ".format(df.AgeCategory.mean()))
    print("Length of stay  {:.2f} ".format(df.LOSDays.mean()))
    print("Race            ")
    target_race = df.RaceDescription.unique().tolist()
    for f in target_race:
        print("\t", f, len(df[df.RaceDescription == f]))
    print("Ethnic          ")
    ethnic = df.EthnicGroupDescription.unique().tolist() 
    for f in ethnic:
        print("\t", f, len(df[df.EthnicGroupDescription == f]))
    
def getDistribution(df, posvids, negvids):
    pdf = df[df.VisitID.isin(posvids)]
    ndf = df[df.VisitID.isin(negvids)]
    
    print("* Distribution")
    print("+ Positves: {:.0f}".format(len(posvids)))
    distGALRE(pdf)
    print("\n- Negatives: {:.0f}".format(len(negvids)))
    distGALRE(ndf)
    
    
def right_align(posvids, dynamic, holdoff_size, min_length, debug=False):
    # pd.isnull(dynamic.ShockOnsetTime): nonshock patients
    # dynamic.MinutesFromArrival < (dynamic.ShockOnsetTime - holdoff_size) * 60: shock patients
    
    # NOTE: changed from < to <= 
    # because 1) diagnosis needs measurements on that time and
    #         2) we need the first Shock label
    selected = pd.isnull(dynamic.ShockOnsetTime) | (dynamic.MinutesFromArrival < (dynamic.ShockOnsetTime - holdoff_size) * 60 ) 
    dynamic_new = dynamic.loc[selected, :]
    if debug:
        print("Shock hold off window size: ", holdoff_size)
        print("Dynamic table (only before shock onset or non shock): ", dynamic_new.shape)

    # Shock length and nonshock length
    selected = pd.isnull(dynamic_new.ShockOnsetTime)
    nonshock = dynamic_new.loc[selected, :].groupby(['VisitIdentifier']).size().\
                reset_index(name='NumberOfRecords').sort_values(['NumberOfRecords'], ascending=False)
    #print(nonshock)

    selected = pd.notnull(dynamic_new.ShockOnsetTime)
    shock = dynamic_new.loc[selected, :].groupby(['VisitIdentifier']).size().\
                reset_index(name='NumberOfRecords').sort_values(['NumberOfRecords'], ascending=False).reset_index(drop=True)
    #print(shock)

    # Select from shock patients, make sure the number of records is larger than min_length
    selected= (shock.NumberOfRecords >= min_length)
    shock = shock.loc[selected, :]

    selected = dynamic_new.VisitIdentifier.isin(shock.VisitIdentifier.unique().tolist())
    dynamic_new_2 = dynamic_new.loc[selected]
    if debug:
        print("Number of shock patients")
        print(len(dynamic_new_2.VisitIdentifier.unique().tolist()))
    RecordsNumber = shock.NumberOfRecords.tolist()


    # Select from non shock patients
    i = 0

    for index, row in nonshock.iterrows():
        #print(row['VisitIdentifier'])
        #print(row['NumberOfRecords'])
        if i >= len(RecordsNumber):
            break
        if i < len(RecordsNumber):
            selected = (dynamic_new.VisitIdentifier == row['VisitIdentifier'])
            #print(RecordsNumber[i])
            #print(dynamic_new.loc[selected,:].sort_values("MinutesFromArrival").shape)
            #print(dynamic_new.loc[selected,:].sort_values("MinutesFromArrival")[:RecordsNumber[i]].shape)
            dynamic_new_2= pd.concat([dynamic_new_2, dynamic_new.loc[selected,:].sort_values("MinutesFromArrival")[:RecordsNumber[i]]])
            i += 1
            if debug and i%100 == 0:
                print(i)

    if debug:
        print("Total number of patient")
        print(len(dynamic_new_2.VisitIdentifier.unique()))

        print("Dynamic table (sample from nonshock)")
        print(dynamic_new_2.shape)

    newdir = "data/right_align/"
    if os.path.exists(newdir)==False:
        os.mkdir(newdir)   
        
    # set shock flag on the last event for each truncated sequence
    pdf = dynamic_new_2[dynamic_new_2.VisitIdentifier.isin(posvids)]
    dynamic_new_2.loc[pdf.groupby(['VisitIdentifier']).tail(1).index, 'Shock'] = 1
    dynamic_new_2[dynamic_new_2.columns] = dynamic_new_2[dynamic_new_2.columns].apply(pd.to_numeric, errors='coerce')
         
    dynamic_new_2.to_csv("data/right_align/balanced_"+str(holdoff_size)+"h.csv", index=False)
    
def cutoff(df, pid, posvids, label, holdoff_size, outfile, debug):
    print("Cut-off records after the first septic shock")
    # Create a new column storing septic shock onset time
    selected = (df[label] == 1.0)
    df.loc[selected, 'ShockTime'] = df.loc[selected,'MinutesFromArrival']
    df['ShockTime'] = df.sort_values(['VisitIdentifier', 'MinutesFromArrival']).groupby('VisitIdentifier').ShockTime.ffill()
    df['ShockTime'] = df.sort_values(['VisitIdentifier', 'MinutesFromArrival']).groupby('VisitIdentifier').ShockTime.bfill()
    df['ShockOnsetTime'] = df.groupby(['VisitIdentifier'])['ShockTime'].transform('min') # first shock time
    #df.ShockOnsetTime = df.ShockOnsetTime*1.0/60 # minutes to hours
    del df['ShockTime']
    print(df.ShockOnsetTime.describe())

    # cut-off records after shockOnsetTime for positives
    selected = pd.isnull(df.ShockOnsetTime) | (df.MinutesFromArrival <= (df.ShockOnsetTime - holdoff_size * 60)) 
    df = df.loc[selected, :]
    if debug:
        print("Shock hold off window size: ", holdoff_size)
        print("Dynamic table (only before shock onset or non shock): ", df.shape)
    
    # set shock flag on the last event for each truncated sequence
    pdf = df[df[pid].isin(posvids)]
    df.loc[pdf.groupby(['VisitIdentifier']).tail(1).index, 'Shock'] = 1
    df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')
    
    # for each in ehours:
#         pdf = lp.right_align(posvids, pdf, holdoff_size = each, min_length=min_length, debug=debug)    
    # ndf = df[~df[pid].isin(posvids)]
#     df = pd.concat([pdf, ndf])
    df = excludeOneRowVisits(df,pid)
    df.to_csv(outfile, index=False)
    return df 
    
        
# def setPosShockForTruncated(ehours, posvids):
#     # run 'tbm_rl.py'
#     # set Shock flags for positive visits    
#     for i in ehours:    
#         edf = tl.loaddf("data/right_align/balanced_"+str(i)+"h.csv") # load the proper file
#         pdf = edf[edf.VisitIdentifier.isin(posvids)]
#         edf.loc[pdf.groupby(['VisitIdentifier']).tail(1).index, 'Shock'] = 1
#         edf[edf.columns] = edf[edf.columns].apply(pd.to_numeric, errors='coerce')
#         edf.to_csv("data/right_align/balanced_"+str(i)+"h.csv", index=False)    

# Intersect both pos and both neg in ICD9 & Rules
def intersectLabels(icd9file, posvid, negvid):
    print('* intersect ICD9SepsisShock & ShockFlag')
    idf = pd.read_csv(icd9file, header=0)
    icd9pos = idf.VisitIdentifier.tolist()

    double_posvid = [x for x in posvid if x in icd9pos]
    double_negvid = [x for x in negvid if x not in icd9pos]
    vid_iTrF = [x for x in negvid if x in icd9pos]
    vid_iFrT = [x for x in posvid if x not in icd9pos]

    statcols = ['Rule_T', 'Rule_F', 'total']
    statrows = ['ICD9_T', 'ICD9_F', 'total']
    data = [[len(double_posvid), len(vid_iTrF), len(icd9pos)],
            [len(vid_iFrT),len(double_negvid), len(vid_iFrT)+len(double_negvid)],
            [len(posvid), len(negvid), len(posvid)+len(negvid)]]
    print(pd.DataFrame(data, statrows, statcols))

    return double_posvid, double_negvid

# intersection of ICD9 and Rules
def getIntersectLabels(mdf, icd9file, posvid, negvid, intersect_outfile):
    double_posvid, double_negvid = intersectLabels(icd9file, posvids, negvids)
    df = mdf[mdf.VisitIdentifier.isin(double_posvid + double_negvid)]
    totvids = double_posvids + double_negvids
    posvids, negvids, totvids = statLabels(mdf, 'ShockFlag', 'Intersection')
    df.to_csv(intersect_outfile, index=False)
    return df, posvids, negvids, totvids
    
        
# --------------------------
# Data Statistics

# get statistics for label from the given dataset, df
def statLabels(df, label, title, pid):

    # Visit Level
    posvid = df[df[label] == 1][pid].unique().tolist()
    posvidlen = len(posvid)
   
    totvid = df[pid].unique().tolist()
    totvidlen = len(totvid)
    
    negvid = [x for x in totvid if x not in posvid]
    negvidlen = len(negvid)

    # Event length for positive/negative visits    
    poslen = len(df[df[pid].isin(posvid)])
    neglen = len(df[df[pid].isin(negvid)]) 
    
    statcols = ['Pos', 'Neg','total']
    statrows = ['Visit', 'Event']
    data = [[posvidlen, negvidlen, totvidlen],
            [poslen, neglen, poslen+neglen]]
    
    if title != '':
        print(title)
        print(pd.DataFrame(data,statrows,statcols))
    
    return posvid, negvid, totvid

# get distribution of death 
def statDeath(df, label, pid):
    shockVids = df[df[label] == 1][pid].unique().tolist()
    OFVids = df[df['OrganFailure'] == 1][pid].unique()
    IFMVids = df[df['Inflammation'] == 1][pid].unique()
    IFTVids = df[df['Infection'] == 1][pid].unique()
    deathVids = df[(df.Death == 1)][pid].unique()
    
    deathTotal = len(deathVids) 
    aliveTotal = len(df[pid].unique()) - deathTotal
    
    shockPos = len(df[(df[pid].isin(shockVids))&(df[pid].isin(deathVids))][pid].unique())
    shockNeg = len(df[(df[pid].isin(shockVids))&(~df[pid].isin(deathVids))][pid].unique())
    OFPos =  len(df[(df[pid].isin(OFVids))&(df[pid].isin(deathVids))][pid].unique())
    OFNeg = len(df[(df[pid].isin(OFVids))&(~df[pid].isin(deathVids))][pid].unique())
    IFMPos =  len(df[(df[pid].isin(IFMVids))&(df[pid].isin(deathVids))][pid].unique())
    IFMNeg = len(df[(df[pid].isin(IFMVids))&(~df[pid].isin(deathVids))][pid].unique())
    IFTPos =  len(df[(df[pid].isin(IFTVids))&(df[pid].isin(deathVids))][pid].unique())
    IFTNeg = len(df[(df[pid].isin(IFTVids))&(~df[pid].isin(deathVids))][pid].unique())
    
    print ("                       Death (total: {})     ".format(deathTotal))
    print ("              +       -        Total      Death/PosLabel  PosLabel/Death")
    print ("Septic Shock ", shockPos,"   ",shockNeg, "    ",shockPos + shockNeg,"    ", 
           np.round(shockPos/len(shockVids),3), "         ", np.round(shockPos/deathTotal,3))
    print ("OrganFailure ", OFPos,"   ",OFNeg, "   ", OFPos + OFNeg,"   ", np.round(OFPos/len(OFVids),3), 
          "        ", np.round(OFPos/deathTotal, 3))
    print ("Inflammation ", IFMPos,"   ",IFMNeg, "   ", IFMPos + IFMNeg, "   ",np.round(IFMPos/len(IFMVids),3), 
          "         ", np.round(IFMPos/deathTotal, 3))
    print ("Infection    ", IFTPos,"   ",IFTNeg, "   ", IFTPos + IFTNeg, "   ",np.round(IFTPos/len(IFTVids),3), 
          "        ", np.round(IFTPos/deathTotal, 3))
          
def statDeathAlive(df, label, pid):
    shockVids = df[df[label] == 1][pid].unique().tolist()
    OFVids = df[df['OrganFailure'] == 1][pid].unique()
    IFMVids = df[df['InflammationFlag'] == 1][pid].unique()
    IFTVids = df[df['InfectionFlag'] == 1][pid].unique()
    deathVids = df[(df.Death == 1)][pid].unique()
    
    deathTotal = len(deathVids) 
    aliveTotal = len(df[pid].unique()) - deathTotal
    
    shockPos = len(df[(df.Death == 1) & (df.VisitIdentifier.isin(shockVids))][pid].unique())
    shockNeg = len(df[(df.Death == 1) & (~df.VisitIdentifier.isin(shockVids))][pid].unique())
    OFPos =  len(df[(df.Death == 1) & (df.VisitIdentifier.isin(OFVids))][pid].unique())
    OFNeg = len(df[(df.Death == 1) & (~df.VisitIdentifier.isin(OFVids))][pid].unique())
    IFMPos =  len(df[(df.Death == 1) & (df.VisitIdentifier.isin(IFMVids))][pid].unique())
    IFMNeg = len(df[(df.Death == 1) & (~df.VisitIdentifier.isin(IFMVids))][pid].unique())
    IFTPos =  len(df[(df.Death == 1) & (df.VisitIdentifier.isin(IFTVids))][pid].unique())
    IFTNeg = len(df[(df.Death == 1) & (~df.VisitIdentifier.isin(IFTVids))][pid].unique())
    
    shockPosLive = len(df[(~df[pid].isin(deathVids)) & (df.VisitIdentifier.isin(shockVids))][pid].unique())
    shockNegLive = len(df[(~df[pid].isin(deathVids)) & (~df.VisitIdentifier.isin(shockVids))][pid].unique())
    OFPosLive = len(df[(~df[pid].isin(deathVids)) & (df.VisitIdentifier.isin(OFVids))][pid].unique())
    OFNegLive = len(df[(~df[pid].isin(deathVids)) & (~df.VisitIdentifier.isin(OFVids))][pid].unique())
    IFMPosLive = len(df[(~df[pid].isin(deathVids)) & (df.VisitIdentifier.isin(IFMVids))][pid].unique())
    IFMNegLive = len(df[(~df[pid].isin(deathVids)) & (~df.VisitIdentifier.isin(IFMVids))][pid].unique())
    IFTPosLive = len(df[(~df[pid].isin(deathVids)) & (df.VisitIdentifier.isin(IFTVids))][pid].unique())
    IFTNegLive = len(df[(~df[pid].isin(deathVids)) & (~df.VisitIdentifier.isin(IFTVids))][pid].unique())

    
    print ("                       Death                              Alive")
    print ("                 +     -     Total   Ratio    +        -      Total    Ratio")
    print ("Shock        ", shockPos,"   ",shockNeg, "  ",shockPos + shockNeg,"   ", np.round(shockPos/deathTotal,2),
          "   ", shockPosLive, "   ",shockNegLive, " ", shockPosLive+shockNegLive,"   ", np.round(shockPosLive/aliveTotal,2))
    print ("OrganFailure ", OFPos,"   ",OFNeg, "   ", OFPos + OFNeg,"   ", np.round(OFPos/deathTotal,2),
          "   ", OFPosLive, "  ",OFNegLive," ", OFPosLive + OFNegLive,"   ", np.round(OFPosLive/aliveTotal,2))
    print ("Inflammation ", IFMPos,"   ",IFMNeg, "   ", IFMPos + IFMNeg, "   ",np.round(IFMPos/deathTotal,2),
          "   ", IFMPosLive, "  ",IFMNegLive, "  ", IFMPosLive + IFMNegLive, "   ",np.round(IFMPosLive/aliveTotal,2))
    print ("Infection    ", IFTPos,"   ",IFTNeg, "   ", IFTPos + IFTNeg, "   ",np.round(IFTPos/deathTotal,2), 
          "    ", IFTPosLive, "  ",IFTNegLive,"     ", IFTPosLive + IFTNegLive, "   ",np.round(IFTPosLive/aliveTotal,2))

# Intersect both pos and both neg in ICD9 & Rules
# def intersectLabels(vf, posvid, negvid):
#     print('* intersect ICD9SepsisShock & ShockFlag')
#     icd9Tvid = vf[vf['SepticShockICD9']==1].VisitIdentifier.tolist()
#     icd9Fvid = vf[vf['SepticShockICD9']!=1].VisitIdentifier.tolist()
# 
#     double_posvid = [x for x in posvid if x in icd9Tvid]
#     double_negvid = [x for x in negvid if x in icd9Fvid]
#     vid_iTrF = [x for x in negvid if x in icd9Tvid]
#     vid_iFrT = [x for x in posvid if x in icd9Fvid]
# 
#     statcols = ['Rule_T', 'Rule_F', 'total']
#     statrows = ['ICD9_T', 'ICD9_F', 'total']
#     data = [[len(double_posvid), len(vid_iTrF), len(icd9Tvid)],
#             [len(vid_iFrT),len(double_negvid), len(icd9Fvid)],
#             [len(posvid), len(negvid), len(posvid)+len(negvid)]]
#     print(pd.DataFrame(data, statrows, statcols))
#     return double_posvid, double_negvid

# Exclude visits(trajectories) with only one row(observation)
# since RL requires 'S, A, R, S' (at least two rows for each trajectories) 
def excludeOneRowVisits(df, pid):
    tf = df.groupby(pid).size().reset_index(name='NumOfRecords').sort_values(['NumOfRecords'], ascending=False).reset_index(drop=True)
    totExNum  = len(tf[tf.NumOfRecords == 1])
    exPosShock = len(df[df[pid].isin(tf[tf.NumOfRecords == 1][pid])].loc[df.Shock == 1])
    print("Inital visits num: ", len(df[pid].unique()))
    print("Total excluded visits with only one row: ", totExNum)
    print(" - Excluded shock positives: ", exPosShock )
    print(" - Excluded shock negatives: ",totExNum - exPosShock )

    df = df.drop(df[df[pid].isin(tf[tf.NumOfRecords == 1][pid])].index)
    
    print("Final visits num: ", len(df[pid].unique()))
    return df
    
    

# ---------------------------------------------------------------- 
#  Preprocessing
# ----------------------------------------------------------------

def prerocessing(df, outfile):
    print("-------------\n Preprocessing\n -------------")
    df = ppDrugs(df, lf.drugs)
    df = ppNewLocationTypeCode(df)
    df = ppOxygenSource(df)
    df = ppChangeToOxygenSource(df)
    df = ppCultures(df, lf.cultures)
    df = ppVerbalGCS(df)
    tl.savedf_date(df, outfile) 
    # Sort
#     df = df[idfeat+feat+labels] 
#     df = df.sort_values(['VisitIdentifier', 'MinutesFromArrival'])
    return df

def ppVerbalGCS(df):
    print("* VerbalGCS")
    # UPDATE yj_MVE SET VerbalGCS = 1 WHERE (VerbalGCS IS NOT NULL) AND (VerbalGCS != 'Oriented');
    # UPDATE yj_MVE SET VerbalGCS = 0 WHERE VerbalGCS = 'Oriented';
    df.loc[df[pd.notnull(df.VerbalGCS) & (df.VerbalGCS != 'Oriented')].index, 'VerbalGCS'] = 1
    df.loc[df[df.VerbalGCS == 'Oriented'].index, 'VerbalGCS'] = 0
    for i in range(2):
        print(' ', i, len(df[df.VerbalGCS == i]) )
    return df

def ppDrugs(df, drugs):
    print("* Drugs")
    for b in drugs:
        df.loc[df[(pd.notnull(df[b]))].index, [b]] = 1
        print(b, len(df[(pd.notnull(df[b]))].index))
    return df

def ppNewLocationTypeCode(df):
    print("* NewLocationTypeCode")
    # 3. NewLocationTypeCode: 'ED', nan, 'ICU   ', 'NURSE ', 'STEPDN', 'Nurse ']
    df.loc[df[(df.NewLocationTypeCode=='NURSE ')|(df.NewLocationTypeCode=='Nurse ')|(df.NewLocationTypeCode=='OUTP')].index, 'NewLocationTypeCode'] = 0
    df.loc[df[df.NewLocationTypeCode=='STEPDN'].index, 'NewLocationTypeCode'] = 1
    df.loc[df[df.NewLocationTypeCode=='ED'].index, 'NewLocationTypeCode'] = 2
    df.loc[df[df.NewLocationTypeCode=='ICU   '].index, 'NewLocationTypeCode'] = 3
    
    for i in range(4):
        print(' ', i, len(df[df.NewLocationTypeCode == i]) )
    return df

def ppOxygenSource(df):
    print("* OxygenSource")
    tdf = df[pd.notnull(df.OxygenSource)] #df['OxygenSource'] = df['OxygenSource'].replace(np.nan, '',regex =True) 
    idx1 = tdf[tdf.OxygenSource.str.contains('trach') | tdf.OxygenSource.str.contains('PAP') | 
           tdf.OxygenSource.str.contains('Vent') | tdf.OxygenSource.str.contains('vent')]['OxygenSource'].index
    df.loc[idx1, ['OxygenSource']]= 1
    idx0=[i for i in tdf.index if i not in idx1]
    df.loc[idx0, ['OxygenSource']]= 0
    print(" 1:{0}, 0:{1}".format(len(df[df.OxygenSource ==1]), len(df[df.OxygenSource ==0])))
    return df
    
def ppChangeToOxygenSource(df):
    print("* ChangeToOxygenSource")
    tdf = df[pd.notnull(df.ChangeToOxygenSource)]
    idx1 = tdf[tdf.ChangeToOxygenSource.str.contains('trach')|tdf.ChangeToOxygenSource.str.contains('PAP') | 
           tdf.ChangeToOxygenSource.str.contains('Vent')|tdf.ChangeToOxygenSource.str.contains('vent')]['ChangeToOxygenSource'].index
    df.loc[idx1, ['ChangeToOxygenSource']]= 1
    idx0=[i for i in tdf.index if i not in idx1]
    df.loc[idx0, ['ChangeToOxygenSource']]= 0
    print(" 1:{0}, 0:{1}".format(len(df[df.ChangeToOxygenSource ==1]), len(df[df.ChangeToOxygenSource ==0])))
    return df
    

def ppCultures(df, cultures):
    print("* Cultures")
    #cultures = ['AFBCulture','BloodCulture','BodyFluidCulture','BronchialCulture','FungusCulture','PcrFluCulture','RespiratoryVirusCulture','UrineCulture','WoundCulture']
    for f in cultures:
        df.loc[df[(df[f]=='SMEAR POS/CULTURE POS')|(df[f]=='SMEAR /CULTURE POS')|(df[f]=='SMEAR NEG/CULTURE POS')|(df[f]=='CULTURE POS')|(df[f]=='TEST POS')].index, f+'_Culture'] = 1
        df.loc[df[(df[f]=='SMEAR POS/CULTURE NEG')|(df[f]=='SMEAR /CULTURE NEG')|(df[f]=='SMEAR NEG/CULTURE NEG')|(df[f]=='CULTURE NEG')|(df[f]=='TEST NEG')].index, f+'_Culture'] = 0
        df.loc[df[(df[f]=='SMEAR POS/CULTURE POS')|(df[f]=='SMEAR /CULTURE POS')|(df[f]=='SMEAR POS/CULTURE NEG')|(df[f]=='TEST POS')].index, f+'_Smear'] = 1
        df.loc[df[(df[f]=='SMEAR NEG/CULTURE POS')|(df[f]=='SMEAR /CULTURE NEG')|(df[f]=='SMEAR NEG/CULTURE NEG')|(df[f]=='TEST NEG')].index, f+'_Smear'] = 0
    # PcrFluCulture: 'TEST POS'??? Smear or Culture or Both?
    return df

    
def ppPostRRTLocation(df):
    # PostRRTLocation: 'HVIS', 'Intensive Care Unit', 'Location unchanged', 'Stepdown Unit', '0'
    print("* PostRRTLocation")
    df.loc[df[(df.PostRRTLocation=='Location unchanged')|(df.PostRRTLocation=='0')].index, 'PostRRTLocation'] = 0.0
    df.loc[df[(df.PostRRTLocation=='Stepdown Unit')|(df.PostRRTLocation=='1')].index, 'PostRRTLocation'] = 1.0
    df.loc[df[(df.PostRRTLocation=='HVIS')|(df.PostRRTLocation=='2')].index, 'PostRRTLocation'] = 2.0 # HVIS: Heart & Vascular Interventional Services
    df.loc[df[(df.PostRRTLocation=='Intensive Care Unit')|(df.PostRRTLocation=='3')].index, 'PostRRTLocation'] = 3.0
    df.loc[df[(df.PostRRTLocation=='Outpatient to ED')|(df.PostRRTLocation=='Emergency Department')|(df.PostRRTLocation=='4')].index, 'PostRRTLocation'] = 4.0
    for i in range(5):
        print(' ', i, len(df[df.PostRRTLocation == i]) )     
    return df

def add_age_race_gender(df):
    sf = tl.loaddf('//Volumes/squirrelhill.csc.ncsu.edu/sepsis/data/yj_static.csv')
    tsf = sf[sf.VisitID.isin(df.VisitIdentifier.unique())]
    tsf.columns = ['PatientIdentifier', 'VisitIdentifier', 'Gender','Race','Age']
    tsf.loc[tsf[tsf.Gender=='Female'].index, 'Gender'] = 0
    tsf.loc[tsf[tsf.Gender=='Male'].index, 'Gender'] = 1 
    tsf.loc[tsf[tsf.Race=='White'].index, 'Race'] = 0 
    tsf.loc[tsf[tsf.Race=='Black or African American'].index, 'Race'] = 1 
    tsf.loc[tsf[(tsf.Race!=0) & (tsf.Race!=1)].index, 'Race'] = 2 
#   tsf.loc[tsf[(tsf.Race=='Other Race' |(tsf.Race=='Multi Racial')|(tsf.Race=='Pacific Islander')].index, 'Race'] = 1 
#         tsf.loc[tsf[tsf.Race=='Asian'].index, 'Race'] = 1 
    result = pd.merge(df, tsf, how='outer', on=['VisitIdentifier'])
    results = result.drop('PatientIdentifier', axis =1)
    results.to_csv(filepath+'data/preproc/3_1_age_race_gender.csv', index=False)
    return results        

def add_age_race_gender_loop(df):
    sf = tl.loaddf('//Volumes/squirrelhill.csc.ncsu.edu/sepsis/data/yj_age_gender.csv')
    for h in hours:
        df = tl.loaddf(filepath+'data/right_align/shocklabel/Trunc_'+str(h)+'h.csv')
        tsf = sf[sf.VisitID.isin(df.VisitIdentifier.unique())]
        tsf.columns = ['PatientIdentifier', 'VisitIdentifier', 'Gender','Race','Age']
        tsf.loc[tsf[tsf.Gender=='Female'].index, 'Gender'] = 0
        tsf.loc[tsf[tsf.Gender=='Male'].index, 'Gender'] = 1 
        tsf.loc[tsf[tsf.Race=='White'].index, 'Race'] = 0 
        tsf.loc[tsf[tsf.Race=='Black or African American'].index, 'Race'] = 1 
        tsf.loc[tsf[(tsf.Race!=0) & (tsf.Race!=1)].index, 'Race'] = 2 
#   tsf.loc[tsf[(tsf.Race=='Other Race' |(tsf.Race=='Multi Racial')|(tsf.Race=='Pacific Islander')].index, 'Race'] = 1 
#         tsf.loc[tsf[tsf.Race=='Asian'].index, 'Race'] = 1 
        result = pd.merge(df, tsf, how='outer', on=['VisitIdentifier'])
        results = result.drop('PatientIdentifier', axis =1)
        results.to_csv(filepath+'data/right_align/subgroups/gender_race_age/traTrunc_'+str(h)+"h.csv", index=False)
        
    
    
    