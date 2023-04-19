import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyAppDomain import AppDomainFpSimilarity

# load training set data
df_train = pd.read_csv('TrainingSet.csv',index_col='CmpdID')
# load testing set data
df_ext = pd.read_csv('ExternalSet.csv',index_col='CmpdID')


# ---------
# I. using the pairwise similarity based AD 
# ---------
adfs = AppDomainFpSimilarity(df_train, smiCol='neuSmi')
adfs.fpSimilarity_analyze('MACCS_keys')
# calculate similarity matrix (SM) between the external set and the training set
SM_ext = adfs.fpSimilarity_xenoCheck(df_ext,'neuSmi')
# using s_cutoff = 0.85 and N_min = 1
CmpdInADFP = adfs.fpSimilarity_xenoFilter(df_ext,SM_ext,0.85,1)


# ---------
# II. using the activity cliff based AD 
# ---------
from metAppDomain import NSG

#the cutoff values can be changed
for sCutoff in np.arange(0.70,0.99,0.01):
    for cCutoff in np.arange(0.10,1.00,0.05):

        df1 = df_train[['neuSmi','y']]
        df2 = df_ext[['neuSmi','y']]
        df1['src'] = 'Train'
        df2['src'] = 'Test'
        df3 = pd.concat([df1,df2],axis=0)
        df3['dataset'] = 'yellow'
        df3.loc[(df3.src=='Test'),'dataset'] = 'blue'
        
        train_CmpdID_list = df_train.CmpdID.values
        test_CmpdID_list = df_test.CmpdID.values
        
        nsg = NSG(df3, smiCol='neuSmi', yCol='y')
        nsg.calcPairwiseSimilarityWithFp('MACCS_keys')
        stf = 15
        nsg.genNSG(sCutoff=sCutoff,includeSingleton=True)
        nsg.genGss()
        nsg.calcCC()
        dff = nsg.calcSoftLocalDiscontinuityScore2(sCutoff,stf)
        
        nsgview = NSGVisualizer(nsg)
        nsgview.nsg.df_train = nsgview.nsg.df_train.join(dff['softLD|MACCS_keys|{}'.format('%.2f'%sCutoff)])
        nsgview_df = nsgview.nsg.df_train

        nsgview_df['LDS_list'] = nsgview_df['softLD|MACCS_keys|{}'.format('%.2f'%sCutoff)]
        
        df_train_nsgview_df = nsgview_df.drop(index=test_CmpdID_list, inplace=False)
        LDscore_ls = df_train_nsgview_df.LDS_list.values
        CmpdID_ls = df_train_nsgview_df.index
        AC_CmpdID_ls = []
        
        for i, LDS in enumerate(LDscore_ls):
            if LDS > cCutoff:
                AC_CmpdID_ls.append(CmpdID_ls[i])
            else:
                continue
        #print(len(AC_CmpdID_ls))
        
        uList = nsg.filterCCwithNodes(AC_CmpdID_ls)
        for i in range(len(test_CmpdID_list)):
            try:
                uList.remove(test_CmpdID_list[i])
                #print(test_CmpdID_list[i])
                #drop(index=test_CmpdID_list[i], inplace=True)
            except:
                pass
            
        adfs_df_train_ACs = pd.read_csv('../data/all_trainset_dfAct_PubF.csv',index_col='CmpdID')
        adfs_df_train_ACs.drop(uList, inplace=True)

        
        ## remove outliers ##
        adfs = AppDomainFpSimilarity(adfs_df_train_ACs, smiCol='neuSmi')
        adfs.fpSimilarity_analyze('MACCS_keys')
        # calculate similarity matrix (SM) between the external set and the training set
        SM_ext = adfs.fpSimilarity_xenoCheck(remained_df,'neuSmi')
        # using sCutoff = 0.85 and N_min = 1
        CmpdInAD = adfs.fpSimilarity_xenoFilter(remained_df,SM_ext,sCutoff,1)
        
        remained_test_df_AD = remained_df.loc[CmpdInAD]
        remained_df_ADFP-AC = remained_test_df_AD.reset_index(drop=True)
        
        

# ---------
# III. using the structure activity landscape based AD 
# ---------


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics

#

'''
STAGE I Calculate AD Metrics
'''
from metAppDomain import NSG
def rigidWt(x, sCutoff=0.85):
    y = np.ones(shape=x.shape)
    y[x < sCutoff] = 0
    return y
#
def expWt(x, a=10, eps=1e-6):
    # a = 3, Liu et al. JCIM, 2018
    return np.exp(-a*(1-x)/(x + eps))
#
wtFunc1a = rigidWt
kw1a = {'sCutoff':0.65}
wtFunc2a = rigidWt
kw2a = {'sCutoff':0.65}
wtFunc1b = expWt
kw1b = {'a':10}
wtFunc2b = expWt
kw2b = {'a':10}

#import data
df_train = pd.read_csv('all_trainset_dfAct_SubC.csv',index_col = 'CmpdID')
df_ext = pd.read_csv('all_extset_dfAct_SubC.csv',index_col = 'CmpdID')
# NSG
nsg = NSG(df_train,yCol='y',smiCol='neuSmi')
#nsg.calcPairwiseSimilarityWithFp('Morgan(bit)',radius=2,nBits=1024)
nsg.calcPairwiseSimilarityWithFp('MACCS_keys')
dfQTSM = nsg.genQTSM(df_ext,'neuSmi')
df_train = df_train[['neuSmi','y']]
df_ext = df_ext[['neuSmi','y']]
df_ext = df_ext.join(nsg.queryADMetrics(dfQTSM, wtFunc1=wtFunc1a,kw1=kw1a, wtFunc2=wtFunc2a,kw2=kw2a,code='|rigid'))
df_ext = df_ext.join(nsg.queryADMetrics(dfQTSM, wtFunc1=wtFunc1b,kw1=kw1b, wtFunc2=wtFunc2b,kw2=kw2b,code='|exp'))
df_ext.to_csv('dfEx_ADMetrics_Classifier.csv')

'''
STAGE II Evaluate Classifier Performance with ADSAL 
'''

from sklearn import metrics
#set different ρs and IA cutoff values according your requirment
ρsDict = {
'rigid':[ 1, 10, 20, 30, 50, 70, 100],
'exp':[ 0.01,  0.5, 1, 1.5,  3, 5, 7]}

yt = df_ext['y_true']
yprob = df_ext['yExt_probA']
# the threshold value 0.5 can also be changed according your actual requirment
yp = (yprob > 0.5).astype(int)
IAVal_List = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


for code in ['rigid','exp']:
    dfn = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfAUC = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfBA = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfP = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfR = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])

    for densLB in dfAUC.columns:
        for LdUB in dfAUC.index:
            adi = df_ext.index[(df_ext['simiDensity|'+code] >= densLB)&(df_ext['simiWtLD_w|'+code] <= LdUB)]
            dfn.loc[LdUB,densLB] = adi.shape[0]
            try:
                dfAUC.loc[LdUB,densLB] = metrics.roc_auc_score(yt[adi],yprob[adi])
            except:
                dfAUC.loc[LdUB,densLB] = np.nan
            dfBA.loc[LdUB,densLB] = metrics.balanced_accuracy_score(yt[adi],yp[adi])
            dfP.loc[LdUB,densLB] = metrics.precision_score(yt[adi],yp[adi])
            dfR.loc[LdUB,densLB] = metrics.recall_score(yt[adi],yp[adi])

    #print the performance of classifier with within ADSAL on the external validation set
    dfn.to_csv('Classifier_{:s}_AD_n.csv'.format(code))
    dfAUC.to_csv('Classifier{:s}_AD_AUC.csv'.format(code))
    dfBA.to_csv('Classifier{:s}_AD_BA.csv'.format(code))
    dfP.to_csv('Classifier{:s}_AD_P.csv'.format(code))
    dfR.to_csv('Classifier{:s}_AD_R.csv'.format(code))
