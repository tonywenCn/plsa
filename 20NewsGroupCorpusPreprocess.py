from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from cplsa import Plsa
import numpy as np
import time

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'talk.politics.guns', 'rec.sport.baseball']
categories = ['comp.os.ms-windows.misc', 'rec.sport.baseball']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#print twenty_train.data
#count_vect = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(['subject', 'lines', 'edu', 'com']))
count_vect = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X_train_counts = count_vect.fit_transform(twenty_train.data)
print "document count: %d" % (len(twenty_train.data))
print dir(X_train_counts)
X_train_counts = X_train_counts.tocoo()
featureNameDict = count_vect.get_feature_names()

rowId = 2 ** 31
wordCnt = 0 
corpus = [{} for i in xrange(len(twenty_train.data))]
for i, j, cnt in zip(X_train_counts.row, X_train_counts.col, X_train_counts.data):
    #print "row=%d\tcol=%d\tcnt=%d" % (i, j, cnt)
    corpus[i][j] = cnt
    wordCnt += cnt

print "begin to init plsa. document size:%d; avg word per document: %f" % (len(corpus), wordCnt * 1.0 / len(corpus))
begin = time.time()
p = Plsa(corpus, 10)
print "begin to train"
p.train(40)
print "training time:%.2f" % (time.time() - begin)
#p.save('./model/model1.20it.data')
zw = p.getPzw()
for i in xrange(p.getTopics()):
    pzw = np.array(zw[i])
    idx = np.argsort(pzw)[::-1]

    wordArr = []
    topN = 100
    for j in xrange(topN if len(idx) > topN else len(idx)):
        wordArr.append( "%s:%f" % (featureNameDict[idx[j]], pzw[idx[j]]))
    print "topic %d\t %s" % (i, " ".join(wordArr)) 

print dir(count_vect)
#print count_vect.get_feature_names()
