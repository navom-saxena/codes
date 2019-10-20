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

# import random forest regressor from ensamble
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# predicting value for 6.5 level. Putting 6.5 in matrix. Double bracket is matrix, single is vector
y_pred = regressor.predict((np.array([[6.5]])))

# visualizing regression results for higher resolution- arranging X from min to max with 0.01 block because its
# non-continous regression model
X_grid = np.arange(min(X), max(X), 0.01)
# restructure vector to matrix of lenth(X) lines an 1 column
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('decision tree regression high resolution')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
