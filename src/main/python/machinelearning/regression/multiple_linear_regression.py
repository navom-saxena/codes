import numpy as np
import pandas as pd

# preprocessing

# reading dataset
dataset = pd.read_csv(
    "/Users/navomsaxena/codes/src/main/resources/machinelearning/multiple_linear_regression.csv")
print(dataset)

# splitting dataset into independent and dependent variables (X and y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print(X)
print(y)

from sklearn.preprocessing import LabelEncoder

# label encoder encodes categorical data (eg country) to mathematical numbers for ML algorithms

labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

from sklearn.preprocessing import OneHotEncoder

# because there is no relation b/w categorical data in our case (countries), we cant put numbers as
# algorithm will think n1 > n2 or vice versa, so we use onehotencoder to create table of categorical column/data where
# value in specific row in table signifies that data is present for that column

oneHotEncoder_X = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder_X.fit_transform(X).toarray()

# avoiding dummy variable trap
X = X[:, 1:]

# splitting dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train, X_test, y_train, y_test)

# all-in regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting test set results
y_pred = regressor.predict(X_test)

# using ordinary least squares regressor from statsModel
import statsmodels.regression.linear_model as sm

# building optimal model using backward elimination
# prepend ones to make constant in y = b0 + b1x1 + b2x2... equation
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
# take X_opt which has all variables to start with backward elimination
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
summary = regressor_OLS.summary()
print(summary)
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
summary = regressor_OLS.summary()
print(summary)
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
summary = regressor_OLS.summary()
print(summary)
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
summary = regressor_OLS.summary()
print(summary)
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(y, X_opt).fit()
summary = regressor_OLS.summary()
print(summary)
