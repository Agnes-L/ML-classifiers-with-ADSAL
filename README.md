# ML-classifiers-with-ADSAL
**1. Load data**

Data used for modeling in this paper are publicly available can be accessed here. Data files conclude three kinds of table. The first one is 'Total_data_for_model.csv', which are the total dataset for modeling in this study. Users can change the way the data is divided according to their modelling needs. The rest data are training data for cross-validaion and the validation data for extearnal validaion, which can be directly used for the folllowing modeling codes.

**2. Construct Classifiers**

Requirements

python --3.7.10

rdkit --2020.09.1.0

scikit-learn --0.24.2

Modeling

The code 'recipes_classifier construction.py' is used to construct classifiers in the study. Users can load the data files followed by the instruction in the code.

Only one molecular representation and one algorithm are shown in the code, and users can change different molecular representations and algorithms according to needs.

**3. Applicabilty Domain**

Special note: Many thanks to Dr. Zhongyu Wang and Bobo Wang for their contributions to the establishment of these application domain characterization methods!

Requirements

matplotlib  --3.1.3

matplotlib-base  --3.1.3

matplotlib-inline   --0.1.2

pyvis            --0.1.9

Calculation

Three application domain characterisation methods for machine learning models are provided in the code , namely chemical space based, activity cliff based and structure activity landscape-based application domains.

Files are 'pyAppDomain.py' and 'recipes_classifier construction.py'

Users can set different application domain stringency levels according to the instructions in the codes and their own needs, in order to achieve the function of improving the prediction of the model.
