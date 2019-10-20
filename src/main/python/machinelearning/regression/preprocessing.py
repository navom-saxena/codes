import numpy as np
import pandas as pd

# reading dataset
print('reading dataset --')

dataset = pd.read_csv(
    "/Users/navomsaxena/codes/src/main/resources/machinelearning/Data_Preprocessing/Data.csv")
print(dataset)

# splitting dataset into independent and dependent variables (X and y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
print(X)
print(y)

# handling missing events
print('handling missing events --')

from sklearn.impute import SimpleImputer

# updating missing values with mean strategy

imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1: 3])
print(X)

# encoding categorical data
print('encoding categorical data --')

from sklearn.preprocessing import LabelEncoder

# label encoder encodes categorical data (eg country) to mathematical numbers for ML algorithms

labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
print(X)

from sklearn.preprocessing import OneHotEncoder

# because there is no relation b/w categorical data in our case (countries), we cant put numbers as
# algorithm will think n1 > n2 or vice versa, so we use onehotencoder to create table of categorical column/data where
# value in specific row in table signifies that data is present for that column

oneHotEncoder_X = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder_X.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
y = labelEncoder_Y.fit_transform(y)
print(y)

# splitting dataset into training and test set
print('splitting dataset into training and test set --')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train, X_test, y_train, y_test)

# feature scaling
# we need to feature scale to avoid unnecessary deviation of one value is >> than other
print('feature scaling of training and test set --')

from sklearn.preprocessing import StandardScaler

standardScaler_X = StandardScaler()
X_train = standardScaler_X.fit_transform(X_train)
X_test = standardScaler_X.transform(X_test)
print(X_train, X_test)
