import pandas as pd  # over fitting, down RMSE()
import numpy as np
from sklearn.model_selection import train_test_split
r_cols = ['filename', 'label', 'chroma_stft', 'spectral_centroid', 'spectral_bandwidth',  # 0 - 4
          'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7',  # 5 - 13
          'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17',  # 14 - 23
          'mfcc18', 'mfcc19', 'mfcc20', 'artist_id', 'composers_id', 'release_time']  # 24 - 29
full_data = pd.read_csv('train-process-full-concat.csv',
                        sep=',', encoding='latin-1')
full_data_values = np.delete(full_data.values, 0, axis=0)
full_data_values = np.delete(
    full_data_values, [0, 27, 28, 29], axis=1)  # len = 24

# print(full_data.head())
data_train, data_test = train_test_split(
    full_data_values, test_size=0.1, random_state=30)
# data_train = full_data_values
# prepare training data
# i = 0
# ind = [27, 28]
# count = 0
# delete_item = []
# for index in ind :
#     for i in range(data_train.shape[0]) :
#         if len(str(data_train[i][index]).split(',')) != 1 :
#             count += 1
#             lst = str(data_train[i][index]).split(',')
#             for item in lst :
#                 tmpItem = data_train[i]
#                 tmpItem[index] = item
#                 data_train = np.append(data_train, [tmpItem], axis=0)
#             if i not in delete_item : delete_item.append(i)

# data_train = np.delete(data_train, delete_item, axis=0)

# X_train_str = np.concatenate(([[x] for x in data_train[:,0]], data_train[:,2:-1]), axis=1)
# Y_train_str = np.array([data_train[:,1]]).T
X_train_str = np.array(data_train[:, 1:])
Y_train_str = np.array([data_train[:, 0]]).T
X_train = np.array([list(map(float, x)) for x in X_train_str])
Y_train = np.array([list(map(float, x)) for x in Y_train_str])
# print(X_train[0, :])

X_test_str = np.array(data_test[:, 1:])
Y_test_str = np.array([data_test[:, 0]]).T
# X_test_str = np.concatenate(([[x] for x in data_test[:,0]], data_test[:,2:-1]), axis=1)
# Y_test_str = np.array([data_test[:,1]]).T
# i = 0

# for i  in range(28) :
#     for item in X_test_str.T[i] :
#         if len(item.split(",")) != 1 :
#             print("at ", i)
#             break;

# X_test = np.array([list(map(float, x)) for x in X_test_str])
# Y_test = np.array([list(map(float, x)) for x in Y_test_str])
##
one = np.ones((Y_train.shape[0], 1))
Xbar = np.concatenate((one, X_train), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y_train)
w = np.dot(np.linalg.pinv(A), b)
lst_w = w[:, 0]


def predict(item):
    # if len(str(item[27]).split(',')) != 1 or len(str(item[26]).split(',')) != 1 : #is array
    #     tmpArr = np.array([item])
    #     if len(str(item[27]).split(',')) != 1 : print("27777")
    #     else : print("28888")
    #     lst_27 = list(x for x in str(item[27]).split(','))
    #     lst_26 = list(x for x in str(item[26]).split(','))
    #     tmp_item = item
    #     for x in lst_27 :
    #         for y in lst_26 :
    #             tmp_item[27], tmp_item[26] = x, y
    #             tmpArr = np.append(tmpArr, [tmp_item], axis=0)
    #     tmpArr = np.delete(tmpArr, 0, axis=0)
    #     tmpArr = np.array([list(map(float, ind)) for ind in tmpArr])
    #     predict_array = np.array(list(np.dot(lst_w[1:].T, i) + lst_w[0] for i in tmpArr))
    #     pre = np.mean(predict_array)
    #     # print("preee is ", pre)
    #     return pre
    # else :
    # print('1')
    item = np.array(list(map(float, item)))
    # print('2')
    return np.dot(lst_w[1:].T, item) + lst_w[0]


def RMSE(X_data_test, Y_data_test):
    SE = 0
    n_tests = X_data_test.shape[0]
    Y_data_test = np.array(list(map(float, Y_data_test)))
    print(Y_data_test)
    print(Y_data_test.shape)
    for i in range(n_tests):
        pred = predict(X_data_test[i])
        print(i, "prediction is ", pred, "fact is ", Y_data_test[i])
        SE += (pred - Y_data_test[i])**2
        # print(i)
    RMSE = np.sqrt(SE/n_tests)
    return RMSE


print('RMSE is ', RMSE(X_test_str, Y_test_str))
print(X_test_str.shape, X_train_str.shape)
