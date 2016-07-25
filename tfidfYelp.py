##Written by Danielle Nash, Michelle Kim, Jonathan Ho, Ellen Kim
## May 12, 2016

import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.ml.feature import Tokenizer
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
import re
import sys
import itertools
import random
data = dict()
topWords = []

def mergeSort(array, indices):
    if len(array)>1:
        mid = len(array)//2
        lefthalf = array[:mid]
        indicesL = indices[:mid]
        righthalf = array[mid:]
        indicesR = indices[mid:]
        mergeSort(lefthalf, indicesL)
        mergeSort(righthalf, indicesR)
        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                array[k]=lefthalf[i]
                indices[k] = indicesL[i]
                i=i+1
            else:
                array[k]=righthalf[j]
                indices[k] = indicesR[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            array[k]=lefthalf[i]
            indices[k] = indicesL[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            array[k]=righthalf[j]
            indices[k] = indicesR[j]
            j=j+1
            k=k+1

def sort(array, indices):
    mergeSort(array, indices)
    array.reverse()
    indices.reverse()

def top5(x):
	indexes1 = x.indices
	values1 = x.values
	indexes2 = indexes1.tolist()
	values2 = values1.tolist()
	inds = []
	for i in range(0, len(indexes2)):
		inds.append(i)
	sort(values2, inds)
	print " "
	for i in range(0,5):
		print data.get(indexes2[inds[i]]), " score: ", values2[i]
		
def top5b(x):
	indexes1 = x.indices
	values1 = x.values
	indexes2 = indexes1.tolist()
	values2 = values1.tolist()
	inds = []
	for i in range(0, len(indexes2)):
		inds.append(i)
	sort(values2, inds)
	topWords = []
	upperBound = 5
	if(len(indexes2) < 5):
		upperBound = len(indexes2)
	for i in range(0, upperBound):
		string1 = str(values2[i]) + "::"+ str(data.get(indexes2[inds[i]]))
		topWords.append(string1)
	return topWords

def makeDict(x):
	global data
	hash = HashingTF(100000)
	for i in range(len(x)):
		word = x[i]
		ind = hash.indexOf(word)
		if ind not in data:
			data.update({ind:word})
	print len(data)
			#print data.keys()
	

def cleanWord(L):
	for i in range(0, len(L)):
		L[i] = re.sub(r'[^\w\s]',"",L[i]).lower()
	#for i in range(0, len(L)):
		#L[i] = re.sub()
	return L
	
def returnTFIDF(tokens, hashingTF):
	tf = hashingTF.transform(tokens)
	idf = IDF(minDocFreq=25).fit(tf)
	tfidf = idf.transform(tf)
	return tfidf
	
#print data
conf = SparkConf().setAppName("TFIDF").set("spark.executor.memory", "2g") 
sc = SparkContext(conf=conf)

# Load documents (one per line).
documentsG = sc.textFile("/Users/daniellenash/Downloads/cleangoodreviews.txt").filter(lambda x : len(x) > 15)
documentsB = sc.textFile("/Users/daniellenash/Downloads/cleanbadreviews.txt").filter(lambda x : len(x) > 15)

docTokensG = documentsG.map(lambda x: x.split(" ")).map(cleanWord)
docTokensB = documentsB.map(lambda x: x.split(" ")).map(cleanWord)

hashingTF = HashingTF(100000)

wordList1 = docTokensB.reduce(lambda a,b : a+b) 
wordList2 = docTokensG.reduce(lambda a,b : a+b)


for word in set(wordList2):
	ind = hashingTF.indexOf(word)
	if ind not in data:
		data[ind] =  word


tfidfB = returnTFIDF(docTokensB, hashingTF)
tfidfG = returnTFIDF(docTokensG, hashingTF)

# 0 is bad, 1 is good
dataB = tfidfB.map(lambda x: LabeledPoint(0, x))
dataG = tfidfG.map(lambda x: LabeledPoint(1, x))

fullData = dataB.union(dataG)

model = LogisticRegressionWithLBFGS.train(fullData, iterations=100)

model.save(sc, "/Users/daniellenash/Downloads/spark-1.6.1-bin-hadoop2.6/python/LRModel")


#To Make Dictionaries
tfidfTop5 = tfidfG.map(lambda x: top5b(x))

print tfidfTop5.take(1)


badReviewsDict = dict()
allItems = tfidfTop5.reduce(lambda a,b: a + b)


for item in allItems:
	t = item.split("::")
	if t[1] not in badReviewsDict:
		badReviewsDict[t[1]] = float(t[0])
	else:
		temp = float(badReviewsDict.get(t[1]))
		avg = (temp + float(t[0]))/2
		badReviewsDict[t[1]] = avg
		
vals = list(badReviewsDict.values())
keys = list(badReviewsDict.keys())
pracInds = []
for i in range (0, len(vals)):
	pracInds.append(i)

sort(vals, pracInds)

print "Top Words"

file = open("goodDict2", 'w')

for i in range (0, len(vals)):
	file.write(" " + str(i)+ "  " + str(keys[pracInds[i]]) +  " : "+ str(vals[i]))
	file.write("\n")



sc.stop()



