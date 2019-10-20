import pandas as pd

# preprocessing

# reading dataset
dataset = pd.read_csv(
    "/Users/navomsaxena/codes/src/main/resources/machinelearning/apriori.csv")
print(dataset)

# extracting as list of list of transactions for apriori. Each transaction is a list of items
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# training Apriori on dataset
from python.machinelearning.associationlearning.apyori import apriori

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# getting results
results = list(rules)
