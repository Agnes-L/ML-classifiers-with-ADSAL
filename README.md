# ML-classifiers-with-ADSAL
1.Load data
Data files conclude three kinds of table. The first one is 'Total_data_for_model.csv', which are the total dataset for modeling in this study. The rest data are training data for cross-validaion and the validation data for extearnal validaion, respectively. 

2.Construct Classifiers
1) Environment Set
python --3.7.10 
rdkit --2020.09.1.0
scikit-learn --0.24.2
2) Modeling 
The code 'recipes_classifier construction.py' is used to construct classifiers in the study. Users can load the data files followed by the instruction in the code.
Only one molecular representation and one algorithm are shown in the code, and users can change different molecular representations and algorithms according to needs.

3. Applicabilty Domain
1) Environment Set
matplotlib  --3.1.3
matplotlib-base  --3.1.3
matplotlib-inline   --0.1.2
pyvis            --0.1.9
2) Calculation
Three application domain characterisation methods are provided in the code for machine learning models, namely chemical space based, activity cliff based and structure activity landscape-based application domains.
Files are 'pyAppDomain.py' and 'recipes_classifier construction.py'
Special note: Many thanks to Dr. Zhongyu Wang and Bobo Wang for their contributions to the establishment of these application domain characterization methods!
Users can set different application domain stringency levels according to the instructions in the codes and their own needs, in order to achieve the function of improving the prediction of the model.
