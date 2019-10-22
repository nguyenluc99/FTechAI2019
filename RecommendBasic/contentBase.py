import datetime
start = datetime.datetime.now()
import numpy as np
import math
import pandas as pd
import sys
from sklearn.linear_model import Ridge
from sklearn import linear_model
np.set_printoptions(precision=2) 

#Reading user file:
u_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('../ml100k/u.user', sep='|', names=u_cols, encoding='latin-1')
n_users = users.shape[0]

i_cols = ['movie_id', 'title', 'release_date', 'date', 'url', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('../ml100k/u.item', sep='|', names=i_cols, encoding='latin-1')
item_train = items.values[:, -19:]

r_cols = ['u_id', 'm_id', 'rating', 'time_stamp']
rating_train = pd.read_csv('../ml100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
rating_test = pd.read_csv('./../ml100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
rate_train = rating_train.values
rate_test = rating_test.values

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(item_train.tolist()).toarray()
print('-----', item_train[3])
# print('-----', item_train.shape)
print('-----', tfidf[3])
# print('-----', tfidf.shape)

def get_rated_item_by_user(rate_matrix, user_id) :
    y = rate_matrix[:,0]
    ids = np.where(y == user_id +1)[0] 
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0 
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)

d = tfidf.shape[1]
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))
# def train(numPart) :
#     d = tfidf.shape[1]
#     W = np.zeros((d, n_users))
#     b = np.zeros((1, n_users))
for n in range(int(n_users)) :
    ids, scores = get_rated_item_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept=True)
    Xhat = tfidf[ids, :]
    clf.fit(Xhat, scores)
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_
 
Yhat = tfidf.dot(W) + b
# print('type of Yhat ', type(Yhat))
n = 3
np.set_printoptions(precision=2) # 2 digits after . 
ids, scores = get_rated_item_by_user(rate_test, n)
Yhat[n, ids]
# print('Rated movies ids :', ids )
# print('True ratings     :', scores)
# print('Predicted ratings:', Yhat[ids, n])
def evaluate(Yhat, rate_matrix) :
    se = 0
    cnt = 0
    for n in range(int(n_users)) :
        ids, truthScore = get_rated_item_by_user(rate_matrix, n)
        pred_score = Yhat[ids, n]
        e = truthScore - pred_score
        se += (e**2).sum(axis = 0)
        cnt += e.size
    return math.sqrt(se/cnt)
# for i in range(10, 1, -1) :
#     numParts =  2**i
#     train(numParts)
print('error for training: ', evaluate(Yhat, rate_train), '\nerror for testing: ', evaluate(Yhat, rate_test))