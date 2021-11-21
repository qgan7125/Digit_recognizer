# Files: digit_recognizer.py
# Author: Quan Gan (graduate student)
# Purpose: implementing different models to classify handwritten single digit (0-9).
#          the models are Naive Bayes, Decistion Tree, K-nearest Neighbors, 
#          Support Vector Machine and Convolutional Neural Network
# Dataset: Digit Recognizer Kaggle competition that is the subset of MNIST data.
# Course: INFO 521

# basic libraries
import os
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt

# sklearn libraies
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# tensorflow libraies
from tensorflow.keras.utils import to_categorical

# Global paths 
DATA_ROOT = "../Data"
DATA_TRAIN = os.path.join(DATA_ROOT + os.sep , "train.csv")

# Read data in
def read_data(filepath, delim = ',', n = 1):
    """
    Using genfromtext to read data into a numpy.array.

    :param filepath: string path to data file.
    :param delim: char string for delimiter separating values on a line in data file.
    :param n: number of lines to skip at the beginning of the file.
    """
    return np.genfromtxt(filepath, delimiter=delim, dtype=None, skip_header = n)

# Draw the distribution of data set
def plot_distribution(dataset):
    """
    Plot the distribution of digit categories

    :param dataset: the numpy array that contains the digit labels
    """
    class_count = np.unique(dataset, return_counts=True)
    plt.figure(figsize=(10,7))
    plt.bar(class_count[0], class_count[1])
    plt.xticks(np.arange(0,10, 1))
    plt.title("Distribution of Original Data Set")
    plt.xlabel('Digit Categories')
    plt.ylabel('Count')

    for i in range(len(class_count[0])):
        plt.annotate(class_count[1][i], (i, class_count[1][i] + 100), ha="center", va="center", size = 8)

    plt.savefig("../figures/dataset_distribution.png", format = "png")
    plt.show()

# Draw handwritten digit in the data set
def plot_digit(dataset, n = 1):
    """
    Plot the handwritten digits in the dataset

    :param dataset: the data set of handwritten digit information
    :param n: number of observation to be ploted
    """
    for i in range(n):
        plt.subplot(330 + 1 + i)
        plt.imshow(dataset[::,1:][i].reshape([28, 28]), cmap="gray")
        plt.xticks(np.arange(0, 28, 7))
        plt.yticks(np.arange(0, 28, 7))
    plt.suptitle("First {} handwritten digit plot".format(n))
    plt.show()

# -------------------------------------------------------------------------------------
# TOP LEVEL 
# -------------------------------------------------------------------------------------

if __name__ == '__main__':
    rawData = read_data(DATA_TRAIN)
    plot_digit(rawData, n = 5)