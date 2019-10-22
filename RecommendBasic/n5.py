import numpy as np
# from sklearn.linear_model import Ridge
# n_samples, n_features = 5000, 5
# rng = np.random.RandomState(0)
# # y = rng.randn(n_samples)
# X = rng.randn(n_samples, n_features)
# y = np.sum(X, axis=1)
# print(y.shape)
# clf = Ridge(alpha=1.0)
# clf.fit(X, y) 

lst = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [3, 1], [3, 2], [3, 3], [3, 4]])
# print(np.max(lst[:, 0]))
# lst2 = np.zeros((10, 2))
# print(lst2)
# print(type(lst), lst)
# ids = np.where(lst == 1)[0]#.astype(np.int32)
# print(ids)
# m = np.mean(ids)
# print(m)
# item_ids = lst[ids, 1]
# print(item_ids)

import datetime
def calAve1():
    lst1 = list(range(30000000))
    start1 = datetime.datetime.now()
    ave1 = np.mean(lst1)
    end1 = datetime.datetime.now()
    print('ave1 : ', ave1, 'time1 : ', end1 - start1)

def calAve2():
    lst2 = list(range(30000000))
    start2 = datetime.datetime.now()
    sum = 0
    for num in lst2 :
        sum += num
    ave2 = sum/len(lst2) 
    end2 = datetime.datetime.now()
    print('ave2 : ', ave2, 'time2 : ', end2 - start2)

# from sklearn.metrics.pairwise import cosine_similarity as cs 
# firMax = np.matrix([[1, 2], [3, 4]])
# secMax = np.matrix([[5, 6], [7, 8], [7, 8]])
# sim = cs(firMax, firMax.T)
# # print(sim)

# a = [3, 4, 12]
# print(a)
# from numpy import linalg as la 
# print(la.norm(a))

import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity


def JSD(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """
    p = np.array(p)
    q = np.array(q)
    m = (p + q) / 2
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
    distance = np.sqrt(divergence)
    return distance

a = [[1,2],[1,4]]
b = [[1,3],[7,8]]
# print('a is : ', a)
# print(cosine_similarity(a, a))
# from sklearn.metrics import jaccard_similarity_score
# # print(jaccard_similarity_score(a[0], b[0]))
# arr_a = np.array(a)
# print('====', type(arr_a))
# jac_matrix = [[jaccard_similarity_score(m, n) for m in arr_a] for n in arr_a ]
# print(jac_matrix)
from datetime import datetime
now = datetime.now().timestamp()
print(now)
from sklearn.metrics import jaccard_similarity_score
a = [[1, 2], [3, 4]]
def get_jaccard_similarity_score(a, b) :
    return jaccard_similarity_score(a, b)


jac_matrix = [[get_jaccard_similarity_score(m, n) for m in a] for n in a ]
print(jac_matrix)
