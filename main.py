"""
Run model benchmarks for Bayesian Softmax Classifier on EMNIST dataset
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from emnist import GetEMNIST
from visualize import Visualize
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import BayesSoftmax as bs
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression

def dim_red(X, cap, useIsomap=True):
    """
    Call the dimensionality reduction techniques, default to isomap, PCA otherwise
    :return: Y a n * 'lower_dimensions' matrix retaining original data ordering (not permuted,
    indexes align from original data)
    """
    lower_dimensions = 9
    if useIsomap:
        consider_n_neighbours = 10
        isomap = Isomap(n_components=lower_dimensions, n_neighbors=consider_n_neighbours)
        return isomap.fit_transform(X[:cap])

    pca = PCA(n_components=3, svd_solver='full')
    #pca = PCA(n_components=.9, svd_solver='full')
    pca.fit(X[:cap])
    print("Optimal Components estimated: " + str(pca.n_components_))
    return pca.transform(X[:cap])





"""
Pipeline implemented here **************************************
"""
random_state = 23059
# Cap observations
n = 20000
# Get input training data from EMNIST
#emnist = GetEMNIST( ['t', 'T', 'n', 'N'], n)
emnist = GetEMNIST( ['a', 'b', 'Cc', 'd', 'e', 'A', 'B', 'D', 'E'], n)
Y = dim_red(emnist.flat_X, n, useIsomap=False)
# Comment out to not visualize
vis = Visualize(emnist, Y)
#vis.showGrid(20)
#vis.n_points_closest_p(3, [-3596, 2948, 4178])
#vis.n_points_closest_p(3, [-4330, -3121, -2124])
#vis.n_points_closest_p(3, [6432, -4508, 2959])
#vis.n_points_closest_p(3, [5980, 4607, -1997])


def evaluate_model(model, name, dim_red, train, test, train_lab, test_lab, params=None):
    """
    :param model: the model to train and test
    :param name: string name of the model
    :param dim_red: string the type of dim reduction used
    :return: none  save accuracy scores to file, plot confusion matrix
    """
    model.fit(train, train_lab)
    guess = model.predict(test)
    err = np.mean(guess != test_lab)
    bal = balanced_accuracy_score(test_lab, guess)

    if params != None:
        print("Error -" + name + "- " + ' '.join(params) + " with " + dim_red + " : " + str(err))
        print("Balanced Accuracy -" + name + "- " + ' '.join(params) + " with " + dim_red + " : " + str(bal))
    else:
        print("Error -" + name + "- with " + dim_red + " : " + str(err))
        print("Balanced Accuracy -" + name + "- with " + dim_red + " : " + str(bal))

    conf = plot_confusion_matrix(model, test, test_lab,
                                 display_labels=["A", "B", "Cc", "D", "E", "a", "b", "d", "e"],
                                 #display_labels=["T", "t", "N", "n"],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    plt.title("Confusion Matrix " + name + " with " + dim_red)
    plt.show()
    return

#No dimension Reduction **************************************

print("NO dimension Reduction **************************************")
# Split training data

train, test, train_lab, test_lab = train_test_split(emnist.flat_X, emnist.labels,
                                                    stratify=emnist.labels,
                                                    test_size=0.3,
                                                    random_state=random_state)


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
               params="k="+str(k_models[int(np.argmax(k_errs))][1]))



#PCA dimension Reduction **************************************
print("PCA dimension Reduction **************************************")
# Y is the lower embedding of x in lower_dimensions
Y = dim_red(emnist.flat_X, n, useIsomap=False)

# Split training data

train, test, train_lab, test_lab = train_test_split(Y, emnist.labels,
                                                    stratify=emnist.labels,
                                                    test_size=0.3,
                                                    random_state=random_state)

d_errs = []
d_models = []
for d in range(2, 15):
    # Generic model to test goes here
    model = RandomForestClassifier(max_depth=d, random_state=random_state)
    cv_score = sum(cross_val_score(model, train, train_lab, n_jobs=-1)) / 5
    d_models.append((model, d))
    d_errs.append(cv_score)

evaluate_model(d_models[int(np.argmax(d_acc))][0], "Random Forest", "PCA", train, test, train_lab, test_lab,
               params="max_depth="+str(d_models[int(np.argmax(d_acc))][1]))

evaluate_model(LogisticRegression(random_state=random_state), "Frequentist Softmax", "PCA", train, test, train_lab, test_lab)

# Iterate though knn models
k_errs = []
k_models = []
for k in range(1, 200, 10):
    # Generic model to test goes here
    model = KNeighborsClassifier(n_neighbors=k)
    cv_score = sum(cross_val_score(model, train, train_lab, n_jobs=-1)) / 5
    k_models.append((model, k))
    k_errs.append(cv_score)

evaluate_model(k_models[int(np.argmax(k_errs))][0], "K-NN", "PCA", train, test, train_lab, test_lab,
               params="k="+str(k_models[int(np.argmax(k_errs))][1]))






# ISOMAP dimension Reduction **************************************
print("ISOMAP dimension Reduction **************************************")
# Y is the lower embedding of x in lower_dimensions
Y = dim_red(emnist.flat_X, n, useIsomap=True)

# Split training data

train, test, train_lab, test_lab = train_test_split(Y, emnist.labels,
                                                    stratify=emnist.labels,
                                                    test_size=0.3,
                                                    random_state=random_state)


d_errs = []
d_models = []
for d in range(2, 15):
    # Generic model to test goes here
    model = RandomForestClassifier(max_depth=d, random_state=random_state)
    cv_score = sum(cross_val_score(model, train, train_lab, n_jobs=-1)) / 5
    d_models.append((model, d))
    d_errs.append(cv_score)

evaluate_model(d_models[int(np.argmax(d_acc))][0], "Random Forest", "Isomap", train, test, train_lab, test_lab,
               params="max_depth="+str(d_models[int(np.argmax(d_acc))][1]))

evaluate_model(LogisticRegression(random_state=random_state), "Frequentist Softmax", "Isomap", train, test, train_lab, test_lab)

# Iterate though knn models
k_errs = []
k_models = []
for k in range(1, 200, 10):
    # Generic model to test goes here
    model = KNeighborsClassifier(n_neighbors=k)
    cv_score = sum(cross_val_score(model, train, train_lab, n_jobs=-1)) / 5
    k_models.append((model, k))
    k_errs.append(cv_score)

evaluate_model(k_models[int(np.argmax(k_errs))][0], "K-NN", "Isomap", train, test, train_lab, test_lab,
               params="k="+str(k_models[int(np.argmax(k_errs))][1]))





# BayesSoftmax dimension Reduction **************************************
print("BayesSoftmax dimension Reduction **************************************")
# Y is the lower embedding of x in lower_dimensions
Y = dim_red(emnist.flat_X, n, useIsomap=True)

# Split training data

train, test, train_lab, test_lab = train_test_split(Y, emnist.labels,
                                                    stratify=emnist.labels,
                                                    test_size=0.3,
                                                    random_state=random_state)


#D = {"T":0, "t":1, "N":2, "n":3}
D = {'a':0, 'b':1, 'Cc':2, 'd':3, 'e':4, 'A':5, 'B':6, 'D':7, 'E':8}
model = bs.BayesSoftMaxClassifier(train, np.array([D[i] for i in train_lab]),
                                  numClasses=9, numSim=10000, burnIn=5000, candVar=.008, paramVar=.06)
model.SamplePosterior()


# Evaluation
model.Predict(test)
err = np.mean(model.predictions != np.array([D[i] for i in test_lab]))
bal = balanced_accuracy_score(np.array([D[i] for i in test_lab]), model.predictions)
print("Error in BayesSoftmax: " + str(err))

model.PlotParamTrace(-1)
model.PlotParamDist(0)
model.PlotPredictiveDistribution(0)

