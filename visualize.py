"""
Plot images and dimensionality reduced data points given from PCA and Isomap
"""

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Visualize:

    alphabet = "abcdefghijklmnopqrstuvwxyz"


    def __init__(self, EM, Y):
        """

        :param emnist: the emnist object
        :param Y: the dimension reduced dataset
        """
        self.Y = Y
        self.emnist = EM
        index1, index2, index3 = self.displayAlongdimension(ppd=15)

        """
        # Code that saves the images along 3 primary dimensions in the reduced space
        
        img_merge1 = np.hstack(self.emnist.emnist.data[EM.data_ind[ind]].T for ind in index1)
        img_merge1 = Image.fromarray(img_merge1)
        img_merge1.save('dim_1horizontal.jpg')

        img_merge2 = np.vstack(self.emnist.emnist.data[EM.data_ind[ind]].T for ind in index2)
        img_merge2 = Image.fromarray(img_merge2)
        img_merge2.save('dim_2horizontal.jpg')
        
        #img_merge3 = np.hstack(self.emnist.emnist.data[EM.data_ind[ind]].T for ind in index3)
        #img_merge3 = Image.fromarray(img_merge3)
        #img_merge3.save('dim_3horizontal.jpg')

        for ind in index2:
            plt.imshow(EM.emnist.data[EM.data_ind[ind]].T, cmap="gray")
            plt.title(EM.labels[ind]
                      + " Reduced at: " + str(Y[ind]))
            #plt.show()
        """
        self.show3D()


    def show3D(self):
        # 3D scatterplot in plotly
        # T: red  t:orange  n:yellow  N:blue
        colorD = {'T': 'rgb(198,50,84)', 't': 'rgb(236,100,73)', 'q': 'rgb(53, 183, 121)', 'Q': 'rgb(253, 231, 37)',
                  'n': 'rgb(251,203,90)', 'N': 'rgb(42,162,163)'}
        # 9 Class problem
        colorD = {'A': 'rgb(219,31,72)', 'B': 'rgb(0,67,105)', 'Cc': 'rgb(1,148,154)', 'D': 'rgb(229,221,200)',
                  'E': 'rgb(253,73,160)', 'a': 'rgb(161,106,232)', 'b': 'rgb(180,254,231)', 'd': 'rgb(96,63,139)',
                  'e': 'rgb(10,10,10)'}
        fig = go.Figure(data=[go.Scatter3d(x=self.Y[:,0], y=self.Y[:,1], z=self.Y[:,2],
                                           mode='markers', marker=dict(
                                                                size=5,
                                                                color=[colorD[ self.emnist.labels[i] ] for i in range(len(self.Y))],
                                                                opacity=0.9
                                                            ))])
        fig.show()


    def show2D(self):
        # 2D scatterplot in plotly
        colorD = {'T': 'rgb(68,1,84)', 't': 'rgb(49, 104, 142)', 'q': 'rgb(53, 183, 121)', 'Q': 'rgb(253, 231, 37)',
                  'n': 'rgb(53, 183, 121)', 'N': 'rgb(253, 231, 37)'}
        fig = go.Figure(data=[go.Scatter(x=self.Y[:,0], y=self.Y[:,1],
                                           mode='markers', marker=dict(
        size=8,
        color=[colorD[ self.emnist.labels[i] ] for i in range(len(self.Y))],                # set color to an array/list of desired values
        opacity=0.6
        ))])
        fig.show()


    def dist(self, p1, p2):
        # Euclidean Norm
        return np.sqrt(sum((p1[i] - p2[i])**2 for i in range(min(len(p1), len(p2)))))


    def index_closest(self, p, ignore=[]):
        """
        :param p: vector
        :return: Return the index in Y closest to the vector p
        """
        best_i = -1
        min_sofar = 99999999
        for i in range(len(self.Y)):
            if i in ignore: continue
            d = self.dist(p, self.Y[i])
            if d < min_sofar:
                best_i = i
                min_sofar = d

        return best_i


    def displayAlongdimension(self, ppd, x=0, y=0, z=0):
        """
        :param ppd points per dimension to acquire
        :param x: the vertical line to extract clostest to
        :param y: the horizontal line to extract clostest to
        :return: list of indexes in emnist.data that are closest to given line
        """

        # Points per dimension to visualize

        index_dim1 = []
        index_dim2 = []
        index_dim3 = []
        print("Points found closest to X dim -----------------------")

        for x_val in np.linspace(min(self.Y[:,0]), max(self.Y[:,0]), ppd):
            index_dim1.append(self.index_closest(np.array([x_val, y, z])))

        print("Points found closest to Y dim -----------------------")

        for y_val in np.linspace(min(self.Y[:, 1]), max(self.Y[:, 1]), ppd+1):
            index_dim2.append(self.index_closest(np.array([x, y_val, z])))

        for z_val in np.linspace(min(self.Y[:, 2]), max(self.Y[:, 2]), ppd):
            index_dim3.append(self.index_closest(np.array([x, y, z_val])))

        return index_dim1, index_dim2[1:], index_dim3


    def showGrid(self, max_amount):
        """
        Use PIL to create a horizontal grid of images
        :param max_amount:
        :return:
        """
        img_merge = np.hstack(self.emnist.emnist.data[i].T for i in self.emnist.data_ind[:max_amount])
        img_merge = Image.fromarray(img_merge)
        img_merge.save('horizontal.jpg')
        print(' '.join(self.emnist.labels[j] for j in range(len(self.emnist.labels[:max_amount]))))


    def n_points_closest_p(self, n, p):
        """
        plot the n images closest to point p in reduced space
        :param n:
        :param p:
        :return:
        """
        found = []
        for _ in range(n):

            i = self.index_closest(np.array(p), ignore=found)
            found.append(i)

            plt.imshow(self.emnist.emnist.data[self.emnist.data_ind[i]].T, cmap="gray")
            plt.title(self.emnist.labels[i]
                      + "\n Reduced at: " + str(self.Y[i]))
            plt.show()

            self.Y = np.delete(self.Y, i, 0)
