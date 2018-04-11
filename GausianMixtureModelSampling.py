import numpy as np
import matplotlib.pyplot as plt

# Samples the dataset according to the weights distribution of 3 Gaussian distribution
def GMsample(n,p):
    """
    n=500
    p=[0.2,0.5,0.3]
    Samples the dataset according to the weights distribution of 3 Gaussian distribution
    :return: Sampled data X
    """
    # choose one of 3 Gaussian distribution
    distr_choice = np.random.choice(np.arange(3), size=n, p=p, replace=True)
    unique, counts = np.unique(distr_choice, return_counts=True)
    ##dict(zip(unique, counts))

    # sample from 3 Gaussian distribution
    g0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], counts[0])
    g1 = np.random.multivariate_normal([3, 0], [[1, 0], [0, 1]], counts[1])
    g2 = np.random.multivariate_normal([0, 3], [[1, 0], [0, 1]], counts[2])
    # generate a union data
    x=[]
    for i in range(n):
        if distr_choice[i]==0:
            x.append(g0[0])
            g0=np.delete(g0, 0, 0)
        if distr_choice[i]==1:
            x.append(g1[0])
            g1=np.delete(g1, 0, 0)
        if distr_choice[i]==2:
            x.append(g2[0])
            g2=np.delete(g2, 0, 0)
    return np.vstack(x)
