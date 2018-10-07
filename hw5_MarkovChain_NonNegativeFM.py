################################### Q1 ##########################
"""
# Markov Chains: Ranking 763 College Football Teams #

Rank 763 college football teams based on the scores of every game in the 2017 season.
The data provided in CFB2017 scores.csv contains the result of one game on each line in the format:
Team A index, Team A points, Team B index, Team B points.
"""
# import os
# from numpy.linalg import inv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#############################################################

def indicator(a,b):
    """
    # the indicator using in updating M
    """
    if a>b:
        return 1
    elif a<b:
        return 0
    else:
        return 0.5


##########################################
def get_M(X,n):
    """
    Training data to find M
    Initialize M to a matrix of zeros, updating M after each competetion
    """
    T,_=X.shape # T competetion in total
    M=np.zeros((n,n))
    for t in range(T):
        # the t_th competetion
        # between team "a" earning points "pa" and team "b" earning points "pb"
        a=X[0][t]-1
        b=X[2][t]-1
        pa=X[1][t]
        pb=X[3][t]
        # update M
        M[a][a]=M[a][a]+indicator(pa,pb)+pa/(pa+pb)
        M[b][b]=M[b][b]+indicator(pb,pa)+pb/(pa+pb)
        M[a][b]=M[a][b]+indicator(pa,pb)+pa/(pa+pb)
        M[b][a]=M[b][a]+indicator(pb,pa)+pb/(pa+pb)
    # After processing all=T games,
    # let M be the matrix formed by normalizing the rows of M so they sum to 1.
    Ms=np.sum(M, axis=1)
    for i in range(n):
        M[i]=M[i]/Ms[i]
    M=M.T
    return M
##########################################
# get_stable_state(M)
"""
find the stationary distribution by the first eigenvector
"""
numbda,Q=np.linalg.eig(M)

for i in range(763):
    if (1-numbda[i])< np.exp(-10):
        print(i,numbda[i])
        z=i

q=Q[:,z]
qs=np.sum(q)
w_stable=q/qs

##########################################
# loading the data
# X = pd.read_csv(os.path.join("/Users/cengjianhuan/Documents/Machine Learning/HW5", 'CFB2017_scores.csv'), header=None)
X = pd.read_csv(('CFB2017_scores.csv'), header=None)
# check data
X.head(n=5)

# constants
n = 763 # n teams in total
w=np.zeros((10000,n))
w[0]=np.ones(n)/n # initialize w[0] to the uniform distribution
M=get_M(X,n)

"""
find the stationary distribution by the first eigenvector, failed
I use matlab to computer eigenvector instead

numbda,Q=np.linalg.eig(M)
q=Q[:,1]
q=np.real(q)
qs=np.sum(q)
w_stable=q/qs
"""

w_stable=q/qs


"""
For question 1.a
List the top 25 teams and their corresponding values in wt for t = 10, 100, 1000, 10000.
"""
dist=np.zeros(10001)
dist[0]=sum((w[0]-w_stable))
for t in range(1,10001):
    tmp=w[t-1]*M.T ###
    w[t]=np.sum(tmp, axis=0)
    dist[t]=sum((w[t]-w_stable))

################################# Q2 ############################
"""
Nonnegative matrix factorization:
The data to be used for this problem consists of :
8447 documents from The New York Times.--j
The vocabulary size is 3012 words. --i
Use this data to construct the matrix X: 3012×8447 --Xij
"""
m,n=X.shape
K=25 #Set the rank to 25
T=100 #run for 100 iterations
w=np.ones((10000,n)) #Each value in W and H can be initialized randomly to a positive number, e.g., from a Uniform(1,2) distribution.
h=np.ones((10000,n))
