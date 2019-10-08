def getWordDict(sen) :
    lst = sen.split(' ')
    tmpDic = {}
    for word in lst :
        tmpDic[word] = 0
    for word in lst :
        tmpDic[word] += 1
    return tmpDic
def computeTF(wordDict, bow) :
    tfDic = {}
    bowCount = len(bow)
    for word, count in wordDict.items() :
        tfDic[word] = count/float(bowCount)
    return tfDic

def computeIDF(docList) :
    import math
    tdfDict = {}
    n = len(docList)
    tdfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList :
        for word, val in doc.items() :
            if val > 0 : 
                tdfDict[word] += 1

    for word, val in tdfDict.items() :
        tdfDict[word] = math.log10(n / float(val))
    return tdfDict

def computeTFIDF (tf, idf) :
    tfidf = {}
    for word, val in tf.items() :
        print(word, val)
        tfidf[word] =  val * idf[word]
    return tfidf
sen1 = 'this is my home that is his home'
sen2 = 'this is my girlF he is FA'
bow1 = sen1.split(' ')
bow2 = sen2.split(' ')
wordSet = set(bow1).union(set(bow2))
wordDict1 = dict.fromkeys(wordSet, 0)
wordDict2 = dict.fromkeys(wordSet, 0)
for word in bow1 :
    wordDict1[word] +=1 
for word in bow2 :
    wordDict2[word] +=1 
import pandas as pd 
df = pd.DataFrame([wordDict1, wordDict2])
# print(df)
tfBow1 = computeTF(wordDict1, bow1)
tfBow2 = computeTF(wordDict2, bow2)
# print(tfBow1)
# print(tfBow2)
idf = computeIDF([wordDict1, wordDict2])
tfidf1 = ((computeTFIDF(tfBow1, idf)))
tfidf2 = ((computeTFIDF(tfBow2, idf)))
finDf = pd.DataFrame([tfidf1, tfidf2])
print(finDf)
# print(computeTF(getWordDict(sen1), sen1.split(' ')))
# print(computeIDF(docList))