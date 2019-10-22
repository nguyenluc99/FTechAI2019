# code on Colab
# Make slide, including image illustrating matrix and operation
# compare between 3 kinds of RS, advantages, disadvantages
# Use matrix multiplication to optimize algorithms instead of for-loop
# Others algorithms to optimize similarity between vectors

# import datetime User-user CF, RMSE = 1.0027163166636008
# User-user CF, RMSE = 1.0027163166636008 time elapsed :  0:06:21.957596    
# start = datetime.datetime.now()
import numpy as np
# import math
import pandas as pd
# import sys
from sklearn.linear_model import Ridge
from sklearn import linear_model
np.set_printoptions(precision=2) 
from scipy import sparse 
from sklearn.metrics.pairwise import cosine_similarity
# import scipy.stats
# import threading
from sklearn.metrics import jaccard_similarity_score
from datetime import datetime
start = datetime.now()

# def thread_func() :
#     #
#     #     

# def JSD(p, q):
#     p = np.array(p)
#     q = np.array(q)
#     m = (p + q) / 2
#     divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
#     distance = np.sqrt(divergence)
#     return distance
i = 0
per = 0.01
def get_jaccard_similarity_score(a, b, length):
    global i
    global per
    i += 1
    timePass = datetime.now().timestamp() - start.timestamp()
    # print(datetime.fromtimestamp(timeElapsed * total / i ).strftime("%H : %M : %S"))
    timeElapsed = str(datetime.fromtimestamp(timePass).strftime("%H : %M : %S"))
    timeEstimated = str(datetime.fromtimestamp(timePass * length / i ).strftime("%H : %M : %S")) 
    if (i/length > per) : 
      print('{}, time elapsed : {}, finished at {}/{}, estimated : {}'.format(str((per - 0.01)*100) + '%',timeElapsed, i, length,timeEstimated ))
      per += 0.01
    return jaccard_similarity_score(a, b)

def jaccard_similarity(arr_a):
    arr_a = arr_a.toarray()
    jac_matrix = [[get_jaccard_similarity_score(m, n, len(arr_a)*len(arr_a)) for m in arr_a] for n in arr_a ]
    #print('======', jac_matrix)
    return jac_matrix


def similarity_func(a, b) :
    cos_sim = cosine_similarity(a, b)
    jaccard_sim = jaccard_similarity(a)
    return cos_sim + jaccard_sim


class CF(object) :
    
    def __init__(self, data, k, dist_func = similarity_func, uuCF = 1):
        self.uuCF = uuCF 
        self.data = data if uuCF else data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        self.n_users = int(np.max(self.data[:, 0])) + 1
        self.n_items = int(np.max(self.data[:, 1])) + 1
        # self.S = None
    
    def add(self, new_data) :
        self.data = np.concatenate((self.data, new_data), axis=0)

    def normalize(self):
        users = self.data[:, 0]
        self.Ybar_data = self.data.copy()
        self.mu = np.zeros((self.n_users,))  # a????
        for uid in range(self.n_users) :
            ids = np.where(users == uid)[0]
            item_ids = self.data[ids, 1]
            rating = self.data[ids, 2]
            mean = np.mean(rating) # average 
            if np.isnan(mean) : mean = 0
            self.Ybar_data[ids, 2] = rating - self.mu[uid]
        self.Ybar = sparse.coo_matrix((self.data[:, 2], (self.data[:, 1], self.data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self) :
        # eps = 1e-6
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T) # self.S  = ???

    def refresh(self) :
        self.normalize()
        self.similarity()
    
    def predict(self, u, i, normalized = 1) :
        if not self.uuCF : 
            u, i = i, u
        ids =  np.where(self.data[:, 1] == i)[0].astype(np.int32) # ????????/
        users_rated_i = (self.data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        # print('sim is, ', sim)
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)
        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def recommend(self, u) :
        ids = np.where(self.data[:, 0] == u)[0]
        item_rated_by_u = self.data[ids, 1].tolist()
        recommended_items = []
        for item in range(self.n_items) :
            if item not in item_rated_by_u :
                rating = self.predict(u, item)
                if rating > 0 : recommended_items.append(item)
        return recommended_items
    
    def print_recommended_item(self) :
        if self.uuCF : count = self.n_users
        else : count = self.n_items
        # count = 10
        for n in range(count) :
            recommended = self.recommend(n)
            if self.uuCF : 
                print('recommend items', recommended, 'for user ', n)
            else : print('recommend users', recommended, 'for item ', n)
    
        


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('../ml100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('../ml100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs = CF(rate_train, k = 30, uuCF = 0)
rs.refresh()

n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    pred = rs.predict(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 

RMSE = np.sqrt(SE/n_tests)
end = datetime.now()
print('User-user CF, RMSE =', RMSE, 'time elapsed : ', end - start)