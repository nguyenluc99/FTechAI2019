import numpy as np 

a = np.array([1, 2, 3, 4])
b = [True, False, True]
lst = np.where(a < 3)
# print(lst)
# print(a[lst])
# print(type(lst))
# print(type(a))
# print((a*a).size)
a = np.array([[1, 2, 3], [2, 3, 4]])
b = np.array([[2, 3], [ 3, 2]])
# print(b.dot(a))
# i = 'abc'
# col1 = 'xxx%s'%i
# col2 = 'xxx{}'.format(i)
# print(col1)        
# print(col2)        

# a = {'s1' : 'h1', 's4' : 'h4'}
# ind = 's1'
# if ind in a :
#     print('ind is %s'%a[ind])

# # from sklearn.linear_model import Ridge
# # clf = Ridge(alpha=0.01, fit_intercept  = True)
# # print('clf', clf, type(clf))


# import pandas as pd
# data = {'col1' : [1,2], 'col2' : [3,4]}
# df = pd.DataFrame(data = data, dtype=np.int8)
# # print(df)
# # print(df.dtypes)



# from sklearn.feature_extraction.text import TfidfTransformer
# dataTest = np.array([[2,5,2], [1,2,9], [9,2,5],[6,1,8]])
# transformer = TfidfTransformer(smooth_idf=True, norm ='l1') # type = function
# tfidf = transformer.fit_transform(dataTest.tolist()).toarray() # type = np array
# print('=======', type(tfidf))
# print(tfidf)
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 4, 3])
print(a.tolist() > b.tolist())