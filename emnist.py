"""
Aquire emnist dataset from PyTorch by query
"""
from torchvision.datasets import EMNIST
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class GetEMNIST:

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    # EMNIST has very frustrating labelling in 'bymerge' - this dictionary corrects it
    my_map = {'t': 46, 46: 't', 'r': 45, 45: 'r', 'q': 44, 44: 'q', 'n':43, 43: 'n', 'h':42, 42:'h',
              'g':41, 41:'g', 'f':40, 40:'f', 'e':39, 39:'e', 'd':38, 38:'d', 'b':37, 37:'b',
              'a':36, 36:'a', 'z':35, 35:'z', 'T':29, 29:'T', 'R':27, 27:'R', 'Q':26, 26:'Q',
              'N':23, 23:'N', 'E':14, 14:'E', 'D':13, 13:'D', 'Cc':12, 12:'Cc', 'B':11, 11:'B',
              'A':10, 10:'A'}

    def __init__(self, letters, n_cap):
        """
        Create a data object to house aspects of the data
        :param letters: a tuple of the English letters wanting to plot
        :param n_cap: capped observations
        """
        self.letters = letters

        # Required PyTorch transform for this project
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Fetch the dataset
        self.emnist = EMNIST(root="./data", split='bymerge', download=True)

        # Split
        self.data_ind, self.targ, self.flat_X, self.labels, self.X = self.filter_by_label(n_cap)


    def filter_by_label(self, cap):
        """
        Filter through the entire bymerge set in EMNIST the letters desired

        :return: data_ind: the indexes in emnist.data that are chosen
                 targ: the indexes in emnist.target that are chosen
                 flax_X: the chosen vectors of images flattened
                 labels: the letter labels as characters
                 X: the vanilla images unflattened
        """

        data_ind = []
        targ = []
        flat_X = []
        labels = []
        X = []

        looking_for = [self.my_map[c] for c in self.letters]

        j = 0
        for i in range(len(self.emnist.data)):
            if j >= cap: break
            if self.emnist.targets[i] in looking_for:
                data_ind.append(i)
                targ.append(i)
                flat_X.append(np.array(self.emnist.data[i]).flatten())
                X.append(np.array(self.emnist.data[i]))
                labels.append(self.my_map[int(self.emnist.targets[i])])
                j += 1

        # Print the distribution of labels here:
        len_labels = len(labels)
        print("Number of total labels: " + str(len_labels))
        for l in self.letters:
            count = labels.count(l)
            print("Frequency of " + l + ": " + str(count/len_labels))

        return data_ind, targ, np.array(flat_X), labels, X




