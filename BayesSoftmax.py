import numpy as np
from scipy.stats import chi2
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

class BayesSoftMaxClassifier:

    def __init__(self, X, Y, numClasses, numSim, burnIn, candVar, paramVar = 2, seed = 9999):
        """

        :param X:
        :param Y:
        :param numClasses:
        :param numSim:
        :param burnIn: 0.5 numSim
        :param candVar: sigma^2 governing MVN of metropoliks hastings
        :param paramVar: sigma^2 governing beta coefficients
        :param seed:
        """
        self.Scaler = StandardScaler(copy = False)
        Xstan = self.Scaler.fit_transform(X)
        self.X = np.concatenate((np.ones_like(X[:, 0]).reshape(-1, 1), Xstan), axis = 1)
        self.numRows, self.numCols = self.X.shape
        self.Y = Y
        self.numClasses = numClasses
        self.numSim = numSim
        self.burnIn = burnIn
        self.candVar = candVar
        self.paramVar = paramVar
        self.seed = seed
        self.acceptanceRate = None
        self.parameterDistributions = None
        self.predDistributions = None
        self.predictions = None

    def Kernel(self, Theta):

        """
        X is our design matrix:
          X[:,0] is a column of ones.
          X[:, 1:] is our data, where each predictor is a column vector.
        Y is our response vector.
        Theta is the vector of our parameter estimates:
          Theta[0:K] are our beta estimates for 1st class.
          Theta[K:2K] are our beta estimates for 2nd class.
          .          .         .         .         .
          .          .         .         .         .
          Theta[(k-1)K:K^2] are our beta estimates for Kth class.
          Theta[-1] is our estimate of sigma^2.
        """

        X, Y = self.X, self.Y
        numRows = self.numRows
        sigma2 = Theta[-1]
        Theta = Theta[:-1]
        numClasses = self.numClasses
        BETA = Theta.reshape(numClasses, -1)
        numBeta = BETA.shape[1]
        #Chisquare prior on sigma^2
        pSig_Dens = chi2.pdf(sigma2, df = 50)
        #MVN with diag covariance prior on Beta matrix
        pBeta_Dens = mvn.pdf(Theta, mean = np.zeros(numBeta*numClasses), cov = np.diag([sigma2]*numBeta*numClasses))

        priorTotal = pSig_Dens * pBeta_Dens
        PI = np.empty([numRows, numClasses])

        for i in range(numRows):
            denom = 0
            num = np.empty(numClasses)
            for j, beta in enumerate(BETA):
                num[j] = np.exp(beta.dot(X[i, :]))
                denom += num[j]

            PI[i, :] = num / denom

        Y_Dens = 1
        for i in range(numRows):
            Y_Dens *= PI[i,:][int(Y[i])]

        return priorTotal * Y_Dens

    def SamplePosterior(self):

        X, Y = self.X, self.Y
        numRows, numCols = self.numRows, self.numCols
        numBeta = numCols
        numClasses = self.numClasses
        totalBeta = numBeta*numClasses
        numSim = self.numSim ; burnIn = self.burnIn
        sigma2 = self.candVar ; np.random.seed(self.seed)
        thetaProposed = np.random.multivariate_normal(mean = np.zeros(totalBeta),
                                                      cov = np.diag([sigma2]*totalBeta),
                                                      size = numSim)

        sigmasProposed = np.random.normal(loc = 0, scale = self.paramVar, size = numSim)
        thetaProposed = np.concatenate((thetaProposed, sigmasProposed.reshape(-1, 1)), axis = 1)
        U = np.random.uniform(0, 1, size = numSim)

        ParamDist_Matrix = np.empty([numSim, totalBeta + 1])
        ParamDist_Matrix[0, :] = np.concatenate((np.zeros(totalBeta), np.array([10])))
        numAcceptances = 0

        for i in range(1, numSim):
            currentTheta = thetaProposed[i, :] + ParamDist_Matrix[i - 1, :]
            currentTheta[-1] = np.maximum(currentTheta[-1], 0.001)
            currentTargetDens = self.Kernel(currentTheta)
            previousTargetDens = self.Kernel(ParamDist_Matrix[i - 1, :])
            alpha = currentTargetDens / previousTargetDens
            if U[i] <= alpha:
                numAcceptances += 1
                ParamDist_Matrix[i, :] = currentTheta
            else:
                ParamDist_Matrix[i, :] = ParamDist_Matrix[i - 1, :]

        ParamDist_Matrix = ParamDist_Matrix[burnIn:, :]
        self.acceptanceRate = numAcceptances/numSim * 100
        print(f"Acceptance Rate: {np.round(self.acceptanceRate, 4)}%")

        self.parameterDistributions = ParamDist_Matrix

    def Predict(self, X_test):

        """
        Below is a description of the 3-D matrix "predictions"
            For each test observation there will be a matrix where:
                The n'th row of predictions will be the distribution of probabilities for
                The n'th class, where 1 \leq n \leq numClasses.
        """
        X_test = self.Scaler.transform(X_test)
        X_test = np.concatenate((np.ones_like(X_test[:, 0]).reshape(-1, 1), X_test), axis = 1)
        numTestRows = X_test.shape[0]
        numClasses = self.numClasses
        numSim = self.numSim ; burnIn = self.burnIn
        predictions = np.empty([numTestRows, numClasses, numSim - burnIn])

        for i, theta in enumerate(self.parameterDistributions):

            PI = np.empty([numTestRows, numClasses])
            theta = theta[:-1]
            BETA = theta.reshape(numClasses, -1)

            for j in range(numTestRows):
                denom = 0
                num = np.empty(numClasses)
                for k, beta in enumerate(BETA):
                    num[k] = np.exp(beta.dot(X_test[j, :]) *.1)
                    denom += num[k]

                PI[j, :] = num / denom

            predictions[:, :, i] = PI

        #assert int(np.sum(predictions[0, :, :], axis = 0).sum()) == numSim - burnIn
        self.predDistributions = predictions

        preds = np.empty(predictions.shape[0])
        for i in range(preds.shape[0]):
            preds[i] = np.argmax(np.mean(predictions[i, :, :], axis = 1))

        self.predictions = preds

    def AcquireAccuracy(self, X_test, Y_test):
        self.Predict(X_test)
        acc = accuracy_score(Y_test, self.predictions)
        return acc

    def PlotPredictiveDistribution(self, obsIndex, plotSave = False):

        if self.numClasses <= 10:
            colours = list(mcolors.TABLEAU_COLORS.keys())
        else:
            colours = list(mcolors.CSS4_COLORS.keys())

        colours = colours[:self.numClasses]

        for i in range(self.numClasses):
            plt.hist(self.predDistributions[obsIndex, i, :],
                     bins = 10, color = colours[i], label = ("Class " + str(i)),
                     alpha = .6);

        title = "Posterior Predictive Distribution: Obs " + str(obsIndex + 1)
        plt.title(title, pad = 8, fontsize = 16)
        plt.ylabel("Frequency", labelpad = 8, fontsize = 13)
        plt.xlabel("Predicted Probability", labelpad = 8, fontsize = 13)
        plt.legend()

        if plotSave:
            plt.savefig("PP_Obs" + str(obsIndex + 1) + ".png", dpi = 400, bbox_inches = "tight")

        else:
            plt.show()

    def PlotParamTrace(self, paramIndex):

        start = self.burnIn ; end = self.numSim
        plt.plot(np.arange(start, end, 1), self.parameterDistributions[:, paramIndex])
        plt.show()

    def PlotParamDist(self, paramIndex):

        plt.hist(self.parameterDistributions[:, paramIndex], bins = 20)
        plt.show()