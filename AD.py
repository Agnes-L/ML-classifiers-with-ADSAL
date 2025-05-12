import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from adsal import NSG

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
kw1a = {'sCutoff':0.80}
wtFunc2a = rigidWt
kw2a = {'sCutoff':0.80}
wtFunc1b = expWt
kw1b = {'a':15}
wtFunc2b = expWt
kw2b = {'a':15}

#import data
dfTr = pd.read_csv('TrainingSet.csv',index_col = 'CmpdID')
dfEx = pd.read_csv('ExternalSet_pred.csv',index_col = 'CmpdID')
# NSG
nsg = NSG(dfTr,yCol='y',smiCol='neuSmi')
#nsg.calcPairwiseSimilarityWithFp('Morgan(bit)',radius=2,nBits=1024)
nsg.calcPairwiseSimilarityWithFp('MACCS_keys')
dfQTSM = nsg.genQTSM(dfEx,'neuSmi')
#dfTr = dfTr[['neuSmi','y']]
#dfEx = dfEx[['neuSmi','y_true']]
dfEx = dfEx.join(nsg.queryADMetrics(dfQTSM, wtFunc1=wtFunc1a,kw1=kw1a, wtFunc2=wtFunc2a,kw2=kw2a,code='|rigid'))
dfEx = dfEx.join(nsg.queryADMetrics(dfQTSM, wtFunc1=wtFunc1b,kw1=kw1b, wtFunc2=wtFunc2b,kw2=kw2b,code='|exp'))
dfEx.to_csv('dfEx_ADMetrics.csv')

dfPlot = pd.read_csv('dfEx_ADMetrics.csv')
densDict = {
'rigid':[1, 2, 3, 5, 12, 20],
'exp':[0.15, 0.2, 0.4, 0.8, 2.2, 4.0]}

yt = dfPlot['y_true']
yprob = dfPlot['yExt_probA']
yp = (yprob > 0.5).astype(int)
VValList = [0.05, 0.15, 0.25, 0.35, 0.45, 0.65]

for code in ['rigid','exp']:
    dfn = pd.DataFrame(index=VValList,columns=densDict[code])
    dfAUC = pd.DataFrame(index=VValList,columns=densDict[code])
    dfBA = pd.DataFrame(index=VValList,columns=densDict[code])
    dfP = pd.DataFrame(index=VValList,columns=densDict[code])
    dfR = pd.DataFrame(index=VValList,columns=densDict[code])
    dfLL = pd.DataFrame(index=VValList,columns=densDict[code])
    for densLB in dfAUC.columns:
        for LdUB in dfAUC.index:
            adi = dfPlot.index[(dfPlot['simiDensity|'+code] >= densLB)&(dfPlot['simiWtLD_w|'+code] <= LdUB)]
            dfn.loc[LdUB,densLB] = adi.shape[0]
            try:
                dfAUC.loc[LdUB,densLB] = metrics.roc_auc_score(yt[adi],yprob[adi])
            except:
                dfAUC.loc[LdUB,densLB] = np.nan
            dfBA.loc[LdUB,densLB] = metrics.balanced_accuracy_score(yt[adi],yp[adi])
            dfP.loc[LdUB,densLB] = metrics.precision_score(yt[adi],yp[adi])
            dfR.loc[LdUB,densLB] = metrics.recall_score(yt[adi],yp[adi])
            dfLL.loc[LdUB,densLB] = metrics.log_loss(yt[adi],yprob[adi],labels=[1,0])
            #
    dfn.to_csv('model_{:s}_AD_n.csv'.format(code))
    dfAUC.to_csv('model_{:s}_AD_AUC.csv'.format(code))
    dfBA.to_csv('model_{:s}_AD_BA.csv'.format(code))
    dfP.to_csv('model_{:s}_AD_P.csv'.format(code))
    dfR.to_csv('model_{:s}_AD_R.csv'.format(code))
    dfLL.to_csv('model_{:s}_AD_LL.csv'.format(code))