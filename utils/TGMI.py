# The following code snippet is modified based on the original code by Hua-CM
# Original code: https://github.com/Hua-CM/TGMI

import pandas as pd
from scipy.stats import rankdata, norm
import numpy as np


def p_adjust(p_values):
    p_values = np.array(p_values)
    p_adjusted = p_values * len(p_values) / rankdata(p_values)
    return p_adjusted


def discretize(vector, nbins=0):
    """
    reproduce R package infotheo::discretize
    :param vector: a list of numbers
    :param nbins: nbins, default:len(vector)**(1/3)
    :return: equal frequency discretization number
    """
    if nbins == 0:
        # default
        # nbins = int(len(vector)**(1/3))
        nbins = len(vector) // 2

    quantile = np.array([float(i) / nbins for i in range(nbins + 1)])  # Quantile: K+1 values
    ranks = rankdata(vector) / len(vector)
    vfunc = np.vectorize(lambda x: (quantile >= x).argmax())
    discretized = vfunc(ranks)
    return discretized


def entropy(vector):
    """
    Calculate entropy use discretized values
    :param vector:
    :return:
    """
    count = np.bincount(vector)
    count = count[count != 0]
    probs = count / len(vector)
    return np.sum((-1) * probs * np.log(probs))


def jentropy(*vectors):
    power = 0
    join_vector = np.zeros(len(vectors[0]), dtype=np.int32)
    for vector in vectors:
        join_vector += vector * pow(10, power)
        power += 1
    return entropy(join_vector)


def centropy(xvector: np.array, yvector: np.array):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(X|Y) = H(Y;X) - H(Y)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jentropy(xvector, yvector) - entropy(yvector)


def infomation(xvector: np.array, yvector: np.array):
    """
    Information Gain, I(X;Y) = H(X) + H(Y) - H(Y,X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(xvector) + entropy(yvector) - jentropy(xvector, yvector)


def MIM(xvector, y1vector, y2vector):
    """
    :param xvector: TF's expression discretized value
    :param y1vector: gene1's expression discretized value
    :param y2vector: gene2's expression discretized value
    :return: MIM for TF-gene1-gene2
    """
    # accelerate
    je1 = jentropy(xvector, y1vector, y2vector)
    je2 = jentropy(xvector, y1vector)
    je3 = jentropy(xvector, y2vector)
    je4 = jentropy(y1vector, y2vector)
    s7 = entropy(y1vector) + entropy(y2vector) + entropy(xvector) - je2 - je3 - je4 + je1
    s456 = 3 * je1 - je2 - je3 - je4
    mim = s7 / (s456+1e-20)
    if mim < 0:
        mim = 0
    return mim


def MIM_permutation(xvector, y1vector, y2vector, ex, ey1, ey2, ey1y2):
    # accelerate
    je1 = jentropy(xvector, y1vector, y2vector)
    je2 = jentropy(xvector, y1vector)
    je3 = jentropy(xvector, y2vector)
    je4 = ey1y2
    s7 = ex + ey1 + ey2 - je2 - je3 - je4 + je1
    s456 = 3 * je1 - je2 - je3 - je4
    mim = s7 / (s456+1e-20)
    if mim < 0:
        mim = 0
    return mim


def triple_interaction(xvector, y1vector, y2vector, permutation=1000):
    """
    :param xvector: TF's expression discretized value
    :param y1vector: gene1's expression discretized value
    :param y2vector: gene2's expression discretized value
    :param permutation: permutation numbers
    :return: MIM for TF-gene1-gene2
    """
    mim = MIM(xvector, y1vector, y2vector)
    if mim == 0:
        return 2
    ex = entropy(xvector)
    ey1 = entropy(y1vector)
    ey2 = entropy(y2vector)
    ey1y2 = jentropy(y1vector, y2vector)
    # generate randomsized dataset
    randomzied_dataset = []
    for i in range(permutation):
        tmp_x = np.random.permutation(xvector)
        randomzied_dataset.append(MIM_permutation(tmp_x, y1vector, y2vector, ex, ey1, ey2, ey1y2))
    # calculate P-value
    if np.std(randomzied_dataset) == 0:
    # if np.isclose(0, np.std(randomzied_dataset)):
        z_score = 0
    else:
        z_score = (mim - np.mean(randomzied_dataset)) / (np.std(randomzied_dataset))
    p_val = 1 - norm.cdf(z_score)
    return p_val
