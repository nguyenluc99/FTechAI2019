import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
from datetime import datetime
start = datetime.now()
class MF(object) :

    def __init__(self, data, k, lam = 0.1, Xinit = None, Winit = None, learning_rate = 0.5, max_iter = 1000, print_every = 100, user_base = 1) :
        self.data = data
        self.k = k
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_base = user_base
        self.n_users = int(np.max(data[:, 0])) + 1
        self.n_items = int(np.max(data[:, 1])) + 1
        self.n_ratings = data.shape[0]
        if Xinit is None : 
            self.X = np.random.randn(self.n_items, k)
        else : self.X = Xinit
        if Winit is None : 
            self.W = np.random.randn(k, self.n_users)
        else : self.W = Winit
        self.data_n = self.data.copy()

    def normalize(self) :
        if self.user_base:
            user_col = 0
            item_col = 1
            n_objects = self.n_users
        else: # item bas
            user_col = 1
            item_col = 0 
            n_objects = self.n_items
        users = self.data[:, user_col]
        self.mu = np.zeros((n_objects, ))
        for n in range(n_objects) :
            ids = np.where(users == n)[0].astype(np.int32)
            ratings = self.data_n[ids, 2]
            m = np.mean(ratings)
            if np.isnan(m) : m = 0
            self.mu[n] = m
            self.data_n[ids, 2] = ratings - self.mu[n]
    def loss (self) :
        L = 0
        # self.n_ratings = 10
        for i in range(self.n_ratings) :
            n, m, rating = int(self.data_n[i, 0]), int(self.data_n[i, 1]), int(self.data_n[i, 2])
            L += 0.5*(rating - self.X[m, :].dot(self.W[:, n]))**2
            # print(n, m, rating, L)
        L /= self.n_ratings
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        # print('=====,', L, self.n_ratings)
        return L

    def get_items_rated_by_user(self, uid) :
        ids = np.where(self.data_n[:, 0] == uid)[0]
        item_ids = self.data_n[ids, 1].astype(np.int32)
        ratings = self.data_n[ids, 2]
        return (item_ids, ratings)

    def get_users_who_rate_item(self, iid) :
        ids = np.where(self.data_n[:, 1] == iid)[0]
        user_ids = self.data_n[ids, 0].astype(np.int32)
        ratings = self.data_n[ids, 2]
        return (user_ids, ratings)
    
    def updateX(self) :
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + self.lam*self.X[m, :]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.k,))

    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + self.lam*self.W[:, n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.k,))
    
    def fit(self):
        self.normalize()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.data)
                print ('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)
    def pred(self, u, i) :
        u = int(u)
        i = int(i)
        if self.user_base :
            bias = self.mu[u]
        else : bias = self.mu[1]
        pred = self.X[i, :].dot(self.W[:, u]) + bias
        return 0 if pred < 0 else 5 if pred > 5 else pred

    def evaluate_RMSE(self, rate_test) :
        n_test = rate_test.shape[0]
        SE = 0
        for n in range(n_test) :
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2]) ** 2

        return np.sqrt(SE/n_test)


r_cols = ['uid', 'mid', 'rating', 'unix_timestamp']
# ratings_base = pd.read_csv('../ml100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv('../ml100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

# # rate_train = ratings_base.values
# # rate_test = ratings_test.values
# rate_train = ratings_base.values
# rate_test = ratings_test.values

# rate_test[:, :2] -= 1
# rate_train[:, :2] -= 1
from sklearn.model_selection import train_test_split
ratings = pd.read_csv('../ml-1m/ratings.dat', sep='::', names=r_cols, encoding='latin-1')
rate = ratings.values
rate[:, :2] -= 1
rate_train, rate_test = train_test_split(rate, test_size=0.33, random_state=42) # random_state = 42 ????
print(rate_test.shape, rate_train.shape)

rs = MF(rate_train, k = 2, lam = 0.1, print_every = 2, learning_rate = 2, max_iter = 10, user_base = 0)
rs.fit()
RMSE = rs.evaluate_RMSE(rate_test)
end = datetime.now()
print(' U _ U RMSE : ', RMSE, 'time elapsed : ', end - start)    


