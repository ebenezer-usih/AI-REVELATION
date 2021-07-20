#MULTIPLE LINEAR REGRESSION

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#ENCODING CATEGORICAL VALUES
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE_X=LabelEncoder()
X[:,3]=LE_X.fit_transform(X[:,3])
OHE_X = OneHotEncoder(categorical_features=[3])
X = OHE_X.fit_transform(X).toarray()

#AVOIDING THE DUMMY VARIABLE TRAP
X=X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#FITTING MULTIPLE LINEAR REGRESSION TO TRAINING SET
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#PREDICTING THE TEST RESULT
y_pred=regressor.predict(X_test)

#BUILDING OPTIMAL MODEL USING BACKWARD ELIMINATION
#MANUAL METHOD
import  statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) #THIS IS DONE TO INCLUDE THE CONSTANT IN THE REGRESSION FORMULA
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  #OLS MEANS ORDINARY LEAST SQUARES
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#AUTOMATIC METHOD
#Backward Elimination with p-values only:
def BackwardElimination(x, y, SL):
    import statsmodels.formula.api as sm
    NumVars = len(x[0])
    for i in range(NumVars):
        regressor_OLS = sm.OLS(endog = y, exog = x).fit()
        MaxVar = max(regressor_OLS.pvalues).astype(float)
        if MaxVar > SL:
            for j in range(NumVars - i):
                if(regressor_OLS.pvalues[j] == MaxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = BackwardElimination(X_opt, y, SL)

#Backward Elimination with p-values and Adjusted R Squared
import statsmodels.formula.api as sm
def BackwardElimination(x, y, SL):
    import statsmodels.formula.api as sm
    NumVars = len(x[0]) 
    for i in range(NumVars):
        regressor_OLS_1 = sm.OLS(endog = y, exog = x).fit()
        adjR_before = regressor_OLS_1.rsquared_adj.astype(float)
        MaxVar = max(regressor_OLS_1.pvalues).astype(float)
        if MaxVar > SL:
            for j in range(NumVars - i):
                if(regressor_OLS_1.pvalues[j] == MaxVar):
                    temp = x
                    x = np.delete(x, j, 1)
                    regressor_OLS_2 = sm.OLS(endog = y, exog = x).fit()
                    adjR_after = regressor_OLS_2.rsquared_adj.astype(float)
                    if(adjR_after <= adjR_before):
                        print(regressor_OLS_1.summary())
                        return temp
                    else:
                        continue
    print(regressor_OLS_1.summary())
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = BackwardElimination(X_opt, y, SL)

