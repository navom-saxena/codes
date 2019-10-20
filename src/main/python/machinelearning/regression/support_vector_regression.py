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

# feature scaling
# we need to feature scale to avoid unnecessary deviation of one value is >> than other.
# sklearn svr doesnt do internal scaling
print('feature scaling of training and test set --')

from sklearn.preprocessing import StandardScaler

standardScaler_X = StandardScaler()
standardScaler_Y = StandardScaler()
X = standardScaler_X.fit_transform(X)
y = standardScaler_Y.fit_transform(y.reshape(-1, 1))
print(X, y)

# import svr library and using radial basis kernal algorithm
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# predicting value for 6.5 level. Putting 6.5 in matrix. Double bracket is matrix, single is vector and removing scale
y_pred_scaled = regressor.predict(standardScaler_X.transform(np.array([[6.5]])))
y_pred = standardScaler_Y.inverse_transform(y_pred_scaled)

# visualizing svr regression results
plt.scatter(X, y, color='red')
# putting polynomial regressor and value as X
plt.plot(X, regressor.predict(X), color='blue')
plt.title('svr regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# visualizing regression results for higher resolution- arranging X from min to max with 0.1 block
X_grid = np.arange(min(X), max(X), 0.1)
# restructure vector to matrix of lenth(X) lines an 1 column
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('svr regression high resolution')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
