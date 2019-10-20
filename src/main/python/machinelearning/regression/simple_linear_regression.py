import matplotlib.pyplot as plt
import pandas as pd

# preprocessing

# reading dataset
dataset = pd.read_csv(
    "/Users/navomsaxena/codes/src/main/resources/machinelearning/Regression/Simple_Linear_Regression/Salary_Data.csv")
print(dataset)

# splitting dataset into independent and dependent variables (X and y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
print(X)
print(y)

# splitting dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
print(X_train, X_test, y_train, y_test)

# fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting Test set results
y_pred = regressor.predict(X_test)

# visualizing training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salary vs experience training set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

# visualizing test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salary vs experience test set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

# visualizing test set results and deviation
# there will be no deviation, lines will overlap and blue because its last executed
plt.scatter(X_test, y_pred, color='green')
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salary vs experience deviation set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()
