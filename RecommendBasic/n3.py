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

from sklearn.linear_model import Ridge
clf = Ridge(alpha=0.01, fit_intercept  = True)
print('clf', clf, type(clf))