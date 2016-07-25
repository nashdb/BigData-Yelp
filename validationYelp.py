##Written by Danielle Nash, Michelle Kim, Jonathan Ho, Ellen Kim
## May 12, 2016

from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.ml.feature import Tokenizer
from pyspark.mllib.linalg import Vectors, SparseVector

data = dict()
scores = []
zero=0
one=0

def countZeros(x):
	if x==0:
		zero=zero + 1
	else:
		one=one + 1

conf = SparkConf().setAppName("TFIDF").set("spark.executor.memory", "2g") 
sc = SparkContext(conf=conf)

place = "/Users/daniellenash/Downloads/goodValidation/"
placeAdd = ["goodValidation1.txt","goodValidation2.txt",
"goodValidation3.txt","goodValidation4.txt","goodValidation5.txt","goodValidation6.txt",
"goodValidation7.txt","goodValidation8.txt","goodValidation9.txt","goodValidation10.txt"]

place2 = "/Users/daniellenash/Downloads/badValidation/"
placeAdd2 = ["badValidation1.txt","badValidation2.txt",
"badValidation3.txt","badValidation4.txt","badValidation5.txt","badValidation6.txt",
"badValidation7.txt","badValidation8.txt","badValidation9.txt","badValidation10.txt"]

hashingTF = HashingTF(100000)
model = LogisticRegressionModel.load(sc, "/Users/daniellenash/Downloads/spark-1.6.1-bin-hadoop2.6/python/LRModel")


for c in range(0,10):
	currentPlace = place +""+ placeAdd[c]
	
	documents = sc.textFile(currentPlace).filter(lambda x : len(x) > 15)
	docTokens = documents.map(lambda x: x.split(" "))
	
	tf = hashingTF.transform(docTokens)
	idf = IDF(minDocFreq=5).fit(tf)
	tfidf = idf.transform(tf)
	
	val = model.predict(tfidf)

	mapped = val.map(lambda x: (x,1))
	reduced = mapped.reduceByKey(lambda a,b: a+b)
	
	zero= 0
	one = 0
	
	if len(reduced.lookup(0)) > 0:
		zero = reduced.lookup(0)[0]
	if len(reduced.lookup(1)) > 0:
		one = reduced.lookup(1)[0]
	
	avg = (one*1.0)/(zero+one)

	scores.append(avg)


print scores

average = 0
tot = 0

for l in range(0, len(scores)):
	tot+=scores[l]

tot = (tot/len(scores)) * 1.0

print tot
sc.stop()