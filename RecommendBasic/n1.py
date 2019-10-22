import datetime
start = datetime.datetime.now()
import numpy as np
import math
import pandas as pd
import sys
from sklearn.linear_model import Ridge
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../ml100k/u.user', sep='|', names=u_cols, encoding='latin-1')
n_users = users.shape[0]
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('../ml100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('../ml100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('../ml100k/u.item', sep='|', names=i_cols, encoding='latin-1')

n_items = items.shape[0]

X0 = items.values
X_train_counts = X0[:, -19:]

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2') 
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray() 

def get_items_rated_by_user(rate_matrix, user_id):
    y = rate_matrix[:,0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1 
    # while index in python starts from 0
    ids = np.where(y == user_id +1)[0] 
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0 
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)

def get_items_rated_by_lst_user(rate_matrix, user_ids):
    y = rate_matrix[:,0] # all users
    all_item_ids = []
    for uid in user_ids : 
        ids = np.where(y == uid +1)[0] 
        item_ids = rate_matrix[ids, 1] - 1 # index starts from 0 
        scores = rate_matrix[ids, 2]
        all_item_ids.append([item_ids, scores])
    return all_item_ids

d = tfidf.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

clf = Ridge(alpha=0.01, fit_intercept  = True)
lst_ids_and_score = get_items_rated_by_lst_user(rate_train,range(n_users))
for n in range(n_users):    
    ids, scores = lst_ids_and_score[n]
    Xhat = tfidf[ids, :]
    
    clf.fit(Xhat, scores) 
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_
    

# predicted scores
Yhat = tfidf.dot(W) + b
n = 10
ids, scores = get_items_rated_by_user(rate_test, n)
Yhat[n, ids]

def evaluate(Yhat, rates):
    se = 0
    cnt = 0
    inner_lst_ids_and_score = get_items_rated_by_lst_user(rates,range(n_users))
    for n in range(n_users):
        ids, scores_truth = inner_lst_ids_and_score[n]
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred # an array
        se += (e**2).sum(axis = 0) # sum of elements in array
        cnt += e.size 
    
    return math.sqrt(se/cnt)

print ('RMSE for training:', evaluate(Yhat, rate_train))
print ('RMSE for test    :', evaluate(Yhat, rate_test))
end = datetime.datetime.now()
print("time elapsed : ", end - start)