'''
Creator:     Kael (Zhongyu Wang)
Email:       wzy.kael@gmail.com
Time:        May. 16th, 2019
'''
import pandas as pd
import numpy as np
#
from sklearn import preprocessing
from sklearn.neighbors import DistanceMetric
#
from sklearn.model_selection import cross_val_score, StratifiedKFold
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


#
scalerTypeDict = {
'min-max':preprocessing.MinMaxScaler,
'standard':preprocessing.StandardScaler}
#
class AppDomainX:
    '''Applicability domain based on specific training set X data & preset options
    #
    Supported A.D. are *(_analyze, _xenoCheck, _xenoFilter):
    distance:   based on various type of distance calculated from input features
    [there are 2 strategies for distance metrics: centroid]
    '''
    def __init__(self,dfTrain,XLabels):
        '''to initialize an AppDomain, a df with proper data is needed
        also needed are yCol and XCols that indicate y and X values'''
        self.dfX = dfTrain[XLabels]
        # these labels are kept for extracting external data set columns!
        self.XLabels = XLabels
    def distance_analyze(self,distanceType='Euclidean',scalerType='standard',centroidType='mean',V=None):
        '''distanceType:
            1. "euclidean" (default)
            2. "seuclidean" (weighted V needed)
            3. "mahalanobis"
            4. "manhattan" (aka city-block; slowest, and similar to Euclidean),
            5. "cosine" (distinct from the previous)
        scalerType:
            1. "min-max" scaler: scale range into [0,1]
            2. "standard" scaler: scaled distribution has mean = 0, and sd = 1 (default)
        centroidType:
            1. "mean" (default)
            2. "median"
        '''
        X0 = self.dfX.values.astype(float)
        self.scalerType = scalerType
        used_scaler = scalerTypeDict[scalerType]
        self.scalerAD = used_scaler().fit(X0)
        self.X = self.scalerAD.transform(X0)
        # strategy I
        self.distanceType = distanceType
        if distanceType == 'mahalanobis':
            V = np.cov(self.X, rowvar=False)
            self.used_distances = DistanceMetric.get_metric(distanceType,V=V)
        elif distanceType == 'seuclidean':
            # weighted in a dividing manner
            self.used_distances = DistanceMetric.get_metric(distanceType,V=1/V)
        elif distanceType in ['euclidean','manhattan','cosine']:
            self.used_distances = DistanceMetric.get_metric(distanceType)
        else:
            raise ValueError('Unknown distance type')
        self.DM_trained = self.used_distances.pairwise(self.X)
        # strategy II
        if centroidType == 'mean':
            self.centroid_trained = self.X.mean(axis=0) # one might also wanna use self.X.median(axis=0)
        elif centroidType == 'median':
            self.centroid_trained = self.X.median(axis=0)
        else:
            raise TypeError('Not supported centroidType: '+centroidType)
        self.radiusVec = self.used_distances.pairwise([self.centroid_trained], self.X)[0,:]
    def distanceCentroid_xenoCheck(self, df_xeno):
        '''centroid away distance'''
        Xext = self.scalerAD.transform(df_xeno[self.XLabels].values.astype(float))
        return self.used_distances.pairwise([self.centroid_trained], Xext)[0,:]
    def distanceCentroid_xenoFilter(self, df_xeno, xenoRadiusVec, radiusPerc=100, radiusThres=None):
        '''default radiusPerc=100, equivalent to maximum radius'''
        if radiusThres is None:
            mask = xenoRadiusVec <= np.percentile(self.radiusVec, radiusPerc) # at least one that is not outside the max
        else:
            mask = xenoRadiusVec <= radiusThres
        # return the index that meet the mask condition
        return df_xeno.index[mask]

#
# fingerprints modules implemented in RDKit
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
#
simiMetricDict = {
'Tanimoto':DataStructs.BulkTanimotoSimilarity,
'Dice':DataStructs.BulkDiceSimilarity,
'cosine':DataStructs.BulkCosineSimilarity}
#
fpTypeDict = {
'topology':FingerprintMols.FingerprintMol,
'Morgan(bit)':AllChem.GetMorganFingerprintAsBitVect,
'Morgan(count)':AllChem.GetMorganFingerprint,
'atom_pair(bit)':Pairs.GetAtomPairFingerprintAsBitVect,
'atom_pair(count)':Pairs.GetAtomPairFingerprint,
'top_torsion':Torsions.GetTopologicalTorsionFingerprintAsIntVect,
'MACCS_keys':MACCSkeys.GenMACCSKeys}
#
class AppDomainFpSimilarity:
    '''Molecular fingerprint similarity-based applicability domain manager.
    Notice that fingerprint similarity is intrinsically in a pairwise manner.
    That is, NO average/centroid/reference fingerprint/data_point could be created (meaningful).
    For pairwise applicability domain, there are 2 parameters to tune: a threshold of the metric, and a count of members that meet that threshold.
    I personally recommend the Morgan radius 2 (i.e. ECFP4) with fixed bits (no more than 1024 bits)
    '''
    def __init__(self, dfTrain, smiCol):
        '''a pd.Series of SMILES code will be stored
        # illegal SMILES will raise error'''
        self.sr_smi = dfTrain[smiCol]
        self.ms = [Chem.MolFromSmiles(smi) for smi in self.sr_smi]
    def fpSimilarity_analyze(self, fpType, simiMetric='Tanimoto', **fpOpt): # radius=2, nBits=1024
        '''fpType: fingerprint to be used
        simiMetric: similarity metric to be used
        fpOpt: key words that used along with fpType
        supported fpType:
            'topology': (basic) Daylight-like fingerprint (ExplicitBitVect, 2048 bits)
            'Morgan(bit)': circular/ECFP-like fingerprint, folded into fixed length, radius and nBits are needed (ExplicitBitVect, specified nBits)
            'Morgan(count)': circular/ECFP-like fingerprint, not folded (UIntSparseIntVect, 4294967295 bits)
            'atom_pair(bit)': atom pair fingerprint (SparseBitVect, 8388608 bits)
            'atom_pair(count)': atom pair fingerprint (IntSparseIntVect, 8388608 bits)
            'top_torsion': topological torsion fingerprint (LongSparseIntVect, 68719476735 bits)
            'MACCS_keys': MACCS keys (ExplicitBitVect, 167 bits)
        * According to personal tests, Morgan fingerprints ((radius 2, 1024/less bit) is reasonable both in chemistry intuition and time cost.
        supported simiMetric:
            'Tanimoto' (mostly used, default)
            'Dice' (more suitable for atom pair & topological torsion fingerprints)
            'cosine'
        '''
        self.fpType = fpType
        self.simiMetric=simiMetric
        self.metricMethod = simiMetricDict[simiMetric]
        self.fpMethod = fpTypeDict[fpType]
        self.fps_trained = [self.fpMethod(m,**fpOpt) for m in self.ms]
        # calculated similarityMatrix (SM) might be useful for other use
        self.SM_trained = np.array([self.metricMethod(self.fps_trained[i],self.fps_trained) for i in range(len(self.fps_trained))])
        self.fpOpt = fpOpt
        #
    def fpSimilarity_xenoCheck(self, df_xeno, smiCol):
        '''df_xeno: dataframe that share the 'featuresChosen' with training set
        smiCol: column corresponding to SMILES code (illegal SMILES will raise error)'''
        ms = [Chem.MolFromSmiles(smi) for smi in df_xeno[smiCol]]
        fps = [self.fpMethod(m,**self.fpOpt) for m in ms]
        return np.array([self.metricMethod(fp,self.fps_trained) for fp in fps])
    def fpSimilarity_xenoFilter(self, df_xeno, similarityMatrix, thresSimilar=0.25, nSimilar=1):
        '''all fingerprint type compatible xenoFilter
        df_xeno: dataframe that share the 'featuresChosen' with training set
        similarityMatrix: the Tanimoto similarity matrix returned by AppDomainFpSimilarity.fpMorganTc_xenoCheck()
        thresSimilar(default=0.25): by (>=) which structures 'similar' to training set are identified;
                                    when the similarity is larger, the AD will be stricter.
        nSimilar(default=1): number of compounds that meet the thresSimilar standard;
                             when the number is larger, the AD will be stricter.
        '''
        TanimotoExtBool = similarityMatrix >= thresSimilar
        mask = TanimotoExtBool.sum(axis=1) >= nSimilar
        return df_xeno.index[mask]




#
# =====
# rapid test scripts
# =====
if __name__=='__main__':
    pass#
