# DATA PREPROCESSING

# IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING DATASETS
datasets=pd.read_csv('Data.csv')
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 3].values

#TAKING CARE OF MISSING DATA
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, 1:3]=imputer.fit_transform(X[:, 1:3])

#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#SPLITTING THE DATA INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size = 0.2, random_state = 0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)