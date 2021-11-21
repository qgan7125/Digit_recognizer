# Files: digit_recognizer.py
# Author: Quan Gan (graduate student)
# Purpose: implementing different models to classify handwritten single digit (0-9)
# Dataset: Digit Recognizer Kaggle competition that is the subset of MNIST data.
# Course: INFO 521

# import libraries
import os
import numpy as np
from matplotlib import pyplot as plt

# Global paths 
DATA_ROOT = "Data"
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

print(read_data(DATA_TRAIN).shape)
