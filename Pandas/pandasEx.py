import os
import pandas  as pd
import numpy as np 
import matplotlib.pyplot as plt 

filename = os.getcwd() + '/fer2013/fer2013.csv'
dataFrame = pd.read_csv(filename, encoding='utf-8', sep=',') #  header=None,
def q2() : 
    len1 = []
    len2 = []
    for row in dataFrame[1] : 
        len1.append(len(row.split(' ')))
    for row in dataFrame[2] : 
        len2.append(len(row.split(' ')))
    data = { 'pixels' : len1, 'Usage' : len2}
    df = pd.DataFrame(data, columns = ['pixels', 'Usage'])
    df.plot()

def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def to_float(lst):
    return [float(i) for i in lst]

lstImg = dataFrame['pixels'][ : 7].tolist()
# lstImg = dataFrame['pixels'][ -7 : ].tolist() # for 7 last figure
        
for ind in range(0, 7):
    float_lst = to_float(lstImg[ind].split(' '))
    img_matrix = to_matrix(float_lst, 48)
    plt.subplot(3, 3, ind + 1)
    plt.imshow(img_matrix, cmap='gray', vmin=0, vmax=255)

plt.show()
