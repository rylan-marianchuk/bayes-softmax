"""
Run model benchmarks for Bayesian Softmax Classifier on Iris dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from main import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("CPS1985.csv", header = 0, index_col = 0)
data.reset_index(drop = True, inplace = True)
X = data.iloc[:, :4].copy() ; Y = data.loc[:, "sector"].copy()
X = np.array(X)
for i, fac in enumerate(Y.unique()):
    Y.loc[Y == fac] = i
Y = np.array(Y)

train, test, train_lab, test_lab = train_test_split(X, Y, test_size = .2, random_state = 9999)

random_state = 2000

print("NO dimension Reduction **************************************")
# Split training data
d_acc = []
d_models = []
for d in range(5, 40, 5):
    # Generic model to test goes here
    model = RandomForestClassifier(max_depth=d, random_state=random_state)
    cv_score = sum(cross_val_score(model, train, train_lab, n_jobs=-1)) / 5
    d_models.append((model, d))
    d_acc.append(cv_score)

evaluate_model(d_models[int(np.argmax(d_acc))][0], "Random Forest", "(none)", train, test, train_lab, test_lab,
               params="max_depth="+str(d_models[int(np.argmax(d_acc))][1]))

evaluate_model(LogisticRegression(random_state=random_state), "Frequentist Softmax", "(none)", train, test, train_lab, test_lab)

# Iterate though knn models
k_errs = []
k_models = []
for k in range(1, 200, 10):
    # Generic model to test goes here
    model = KNeighborsClassifier(n_neighbors=k)
    cv_score = sum(cross_val_score(model, train, train_lab, n_jobs=-1)) / 5
    k_models.append((model, k))
    k_errs.append(cv_score)

evaluate_model(k_models[int(np.argmax(k_errs))][0], "K-NN", "(none)", train, test, train_lab, test_lab,
               params="k="+str([int(np.argmax(k_errs))][1]))

