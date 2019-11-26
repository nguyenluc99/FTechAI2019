import numpy as np
# lst = np.array([0,1,2,3,4,5])
# lst2 = np.array([[],[]])
# # print(lst2.shape)
# newlst = np.concatenate(([lst[1]],lst[ 3 : ]))
# # print(newlst)
# arr = [['1.2', '2', '3'], ['1', '2', '3']]
# intt = [list(map(float, x)) for x in arr ]
# print(intt)
a = [1, 2, 3, 4]
b = [1078060306,7,0.28971103,2512.5694384183644,2620.9919497421024,5566.082311441654,0.10636477093456496,-55.074580000000005,79.873375,15.519704999999998,37.209595,-2.4479845,6.0142765,-3.1413496,5.806208,-11.192525999999999,2.7122982,-5.116964,-0.21357735,-5.8451543,-0.7191203,-4.8166533,-0.110485986,-6.232977,-1.3030266000000001,-5.1462617,1.2940463,"67306,602637,432376","100011,7351,100086",'2018-01-04 00:53:00']
# print(b[28])
# for i in range(len(a)) :
#     print(a[i])
#     if i % 2 == 0 : 
        # i -= 1
# for x in a ,b :
#     print(x)


tmp_lst = np.array([['1', '2'], ['3', '4']])
tmp_lst2 = np.array(['1', '2', '4', '5'])
r_cols  = np.array(['filename','label','chroma_stft','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20','artist_id','composers_id','release_time'])
int_lst = np.array(['filename','chroma_stft','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'])
# print(r_cols.shape)
# print(int_lst.shape)
a = [1078365627,3,0.29793018,2077.9190849002407,2313.716443403081,4383.155479011204,0.09154388474167637,-65.88247,106.98664,4.0207286,15.091698999999998,-1.6313189,2.0656164,-4.7894435,-0.6138340999999999,-11.299748,1.2110287,-3.7211062999999993,4.671317,-1.7514745999999999,-0.57969075,-7.2095899999999995,-1.91431,-7.890776,-1.7815105,-7.961927,-4.307783000000001,"7904,9608,721071,721365","102863,2018-11-12 22:58:00"]
print(len(a))
# b = [2, 3, 4]
# for x,y in zip(a, b):
#     print(x * y)