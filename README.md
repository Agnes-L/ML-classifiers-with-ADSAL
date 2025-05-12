# ML-classifiers-with-ADSAL

**1. Load data**

Data used for modeling in this paper are publicly available can be accessed here. Data files conclude three kinds of table. The first is ‘Total_data_for_model.csv’, which is the entire data set used for modelling in this study. 
The user can change the division of the data according to the modelling needs. The rest of the data are training data for cross-validation and validation data for external validation (the external validation set data gives the prediction results for the convenience of the application domain).

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

ADSAL is already available online: https://test.pypi.org/project/adsal/

** Installation steps** 

conda create -n AD python=3.7.10 -y

conda activate AD

conda install -c rdkit rdkit=2020.09.1.0 -y

pip install scikit-learn==0.24.2

pip install matplotlib==3.1.3

conda install matplotlib-base==3.1.3 -y

pip install matplotlib-inline==0.1.2

pip install pyvis==0.1.9

pip install -i https://test.pypi.org/simple/ adsal

Calculation

The code for calculating the application domain indicators is detailed in the file as AD.py.

Users can set different application domain stringency levels according to the instructions in the codes and their own needs, in order to achieve the function of improving the prediction of the model.

Feel free to try and use it!

