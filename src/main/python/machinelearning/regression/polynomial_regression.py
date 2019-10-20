import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# preprocessing

# reading dataset
dataset = pd.read_csv(
    "/Users/navomsaxena/codes/src/main/resources/machinelearning/regression.csv")
print(dataset)

# splitting dataset into independent and dependent variables (X and y)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
print(X)
print(y)

# fitting linear regression to dataset
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# fitting polynomial regression dataset
# creating X_poly which is transformed X that contains degree = 0,1,2,3,4 values of column in X using fit_transform
from sklearn.preprocessing import PolynomialFeatures

poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)
# creating another linear regressor to fit X_poly and y. Hence using new regressor
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)

# visualizing linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('linear regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# visualizing polynomial regression results
plt.scatter(X, y, color='red')
# putting polynomial regressor 2 and value as generic X_poly instead of hardcoding X_poly
plt.plot(X, linear_regressor_2.predict(poly_regressor.fit_transform(X)), color='blue')
plt.title('polynomial regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# visualizing polynomial regression results for higher resolution- arranging X from min to max with 0.1 block
X_grid = np.arange(min(X), max(X), 0.1)
# restructure vector to matrix of lenth(X) lines an 1 column
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear_regressor_2.predict(poly_regressor.fit_transform(X_grid)), color='blue')
plt.title('polynomial regression high resolution')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# predicting new result with linear regression
linear_prediction = linear_regressor.predict([[6.5]])

# predicting new result with polynomial regression
poly_prediction = linear_regressor_2.predict(poly_regressor.fit_transform([[6.5]]))
