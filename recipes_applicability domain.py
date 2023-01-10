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
# II. using the pairwise similarity based AD 
# ---------

start_time = str(time.ctime()).replace(':','-').replace(' ','_')
start = time.time()
cCutoff_list =[]
sCutoff_list =[]
num_remained_list = []
ad_output = []


for sCutoff in np.arange(0.70,0.99,0.01):
    for cCutoff in np.arange(0.10,1.00,0.05):
        remained_df = pd.read_csv('../data/all_extset_dfAct_PubF.csv',index_col='CmpdID')

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
        #nsgview_df_20211209 = nsgview_df_20211209.rename(columns = {"softLD|MACCS_keys|0.85": "LDS_list"})
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