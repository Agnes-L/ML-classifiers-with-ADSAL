

# -----------
# I.  Construcing classifiers, here using MACCS and RandomForest algorithm as an example
# -----------


import os
os.chdir('...')

import pandas as pd
import numpy as np
from numpy import interp 
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold, permutation_test_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn import preprocessing
import pickle

import matplotlib.pyplot as plt


#import data 
dfTrain_fea = pd.read_csv('TrainingSet.csv')
dfExt_fea = pd.read_csv('ExternalSet.csv')
sharedDesc = dfTrain_fea.columns[3:]
y = dfTrain_fea['y'].values
X0 = dfTrain_fea[sharedDesc].values.astype(float)
scaler = preprocessing.StandardScaler()
scaler.fit(X0)
X = scaler.transform(X0)
X1 = dfExt_fea[sharedDesc].values.astype(float)
scaler.fit(X1)
X_test = scaler.transform(X1)

# 5x5 internal CV
parameters = {'n_estimators':[100,300,500,900,1100,1300,1500]}
cv2=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
gridCV = GridSearchCV(RandomForestClassifier(n_estimators=100),parameters,
scoring={'roc_auc':'roc_auc','balanced_accuracy':'balanced_accuracy',
'average_precision':'average_precision','precision':'precision','recall':'recall'},
refit='roc_auc',cv=cv2, return_train_score=True, n_jobs=30)
gridCV.fit(X,y)
pd.DataFrame(gridCV.cv_results_).to_csv('trainset_gridCV10_rf_maccs_5C5g.csv'.format(fea))

#find best parameters 
best_params = gridCV.best_params_
best_score = gridCV.best_score_
clf = gridCV.best_estimator_

#save trainset results
trainset_params = {"fps":[fea],"best_params":[best_params],"best_AUC_score":[best_score]}
trainset_results = pd.DataFrame(trainset_params)

#save optimal model
with open("best_rf_maccs.pickle".format(fea), 'wb') as file:
    pickle.dump(gridCV.best_estimator_ , file)


#extset predict
yExt_true=dfExt_fea['y']
if alg == 'svc':
    yExt_pro = model.decision_function(X_test)
    scaler = MinMaxScaler( )
    scaler.fit(yExt_pro.reshape(-1, 1))
    scaler.data_max_
    yExt_prob=scaler.transform(yExt_pro.reshape(-1, 1))
    yExt_prob = yExt_prob.flatten()
else:
    yExt_pro = model.predict_proba(X_test)
    yExt_prob = yExt_pro[:,1]
yExt_probA = pd.Series(yExt_prob)
yExt_pred = pd.DataFrame(model.predict(X_test))
yExt_conf = pd.DataFrame(np.abs((yExt_prob-0.5)/0.5))
yExt_error = pd.DataFrame(yExt_true-yExt_probA)
dfExt = dfExt_fea[['CmpdID','neuSmi','y']]
pred_results = pd.concat([dfExt,yExt_pred,yExt_probA,yExt_error,yExt_conf],axis =1)
pred_results.columns = ['CmpdID','neuSmi','y_true','y_pred','yExt_probA','err','confidence']
pred_results.to_csv('extset_rf_maccs_noAD.csv')

#extset metrics
AUC =  metrics.roc_auc_score(yExt_true, yExt_prob[:,1])
BA = metrics.balanced_accuracy_score(yExt_true,yExt_pred.astype("int64"))
precision = metrics.precision_score(yExt_true,yExt_pred.astype("int64"), pos_label = 1)
recall = metrics.recall_score(yExt_true,yExt_pred.astype("int64"))
MCC = metrics.matthews_corrcoef(yExt_true,yExt_pred.astype("int64"))
extset_metrics = {"AUC":[AUC], "balanced_accuracy":[BA], "precision":[precision], "recall":[recall]}
extset_results = pd.DataFrame(extset_metrics)

#merge output table
model_outputs = pd.concat([trainset_results,extset_results],axis = 1)
model_outputs.to_csv('rf_maccs_optiModel_results.csv')
