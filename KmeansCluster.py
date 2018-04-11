############################## KmeansClustering ##############################
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

# assign a class ci for a vector xi given k class centers mu
def assign(x,mu,k):
    d=np.linalg.norm(x-mu[0])
    c=0
    for i in range(1,k):
        if d>np.linalg.norm(x-mu[i]):
            d=np.linalg.norm(x-mu[i])
            c=i
    return c,d

# the main updating process of k means clustering
def updating(mu,x,k):
    # initialize the classes ci
    c=np.zeros((n,), dtype=np.int)
    # initialize the counting sum s of each class
    s=np.zeros((k, 2))
    # initialize the counting number cn of data vectors in each cluster
    cn=np.zeros((k,), dtype=np.int)
    # initialize the obejective function
    L=0

    # update all classes ci given each class center
    for i in range(n):
        c[i],d=assign(x[i],mu,k)

        # for updating mu later
        cn[c[i]]=cn[c[i]]+1
        s[c[i]]=s[c[i]]+x[i]
        # for calculating the obejective function
        L=L+d

    # update each class center given all classes ci
    for i in range(k):
        mu[i]=s[i]/cn[i]
        return mu,c,L

############################################################
# the main function
def KmeansClustering(x, k, T):
        """
        data: X, n*d matrix
        the number of clusters: k, a scalar
        the time of iteration: T, a scalar
        return: the value of the K-means objective function
        return: the final clusters of data X
        """
        # Constants
        n,d=x.shape

        # Randomly initialize mu
        index=np.random.choice(np.arange(n), size=k)
        mu=[]
        for i in range(k):
            print(index[i])
            mu.append(x[index[i]])
        mu=np.vstack(mu)

        # Update within T iteration
        for t in range(T):
            mu,c,L[t]=updating(mu,x,k)

        return c, L

#############################################################
