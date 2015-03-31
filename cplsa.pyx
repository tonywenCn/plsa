# -*- coding: utf-8 -*-
import math
import operator
import random
import gzip
import sys
import time
import marshal
import numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.utility cimport pair
cimport numpy as np
import random
from cython.operator cimport dereference as deref, preincrement as inc

cdef void resetMatrix(vector[vector[float]]& m, float value, int randomStat):
    cdef int i 
    for i in xrange(m.size()):
        resetVector(m[i], value, randomStat)

cdef void resetVector(vector[float]& v, float value, int randomStat):
    cdef int i 
    for i in xrange(v.size()):
        if randomStat != 0:
            v[i] = random.random()
        else:
            v[i] = value

cdef class Plsa:
    cdef int topics
    cdef int words
    cdef int docs
    cdef vector[int] each

    # training corpus
    cdef vector[unordered_map[int, int]] corpus

    # likelihood:
    cdef float likelihood

    # decrease factor
    cdef float beta

    # init pzw: p(w|z)
    cdef vector[vector[float]] zw

    # init pdz: p(z|d)
    cdef vector[vector[float]] dz

    # dw_z: p(z|d,w)
    cdef vector[unordered_map[int, vector[float] ] ] dw_z

    # init p_dw: p(d,w)
    cdef vector[unordered_map[int, float]] p_dw

    cdef _init(self, corpus, topics=2):
        self.likelihood = 0.0
        self.beta = 1.0 # 0.8
        self.topics = topics # topic count
        self.words = max(map(lambda x: max(x.keys()), corpus)) + 1 #max(reduce(operator.add, map(lambda x:x.keys(), corpus)))+1
        self.docs = len(corpus) # document count
        print "docs:%d\twords:%d\ttopics:%d" % (self.docs, self.words, self.topics)

        cdef int i, j
        cdef int wordId, wordCnt
        cdef unordered_map[int, int] document
        cdef int documentWords = 0
        cdef float norm = 0.0
        for i in xrange(self.docs):
            self.corpus.push_back(document)
            documentWords = 0
            for wordId, wordCnt in corpus[i].iteritems():
                self.corpus[self.corpus.size() -1][wordId] = wordCnt
                documentWords += wordCnt
            self.each.push_back(documentWords)

        # init pzw: p(w|z)
        self.zw.resize(self.topics)
        for i in xrange(self.topics):
            self.zw[i].resize(self.words)
            norm = 0
            for j in xrange(self.words):
                self.zw[i][j] = random.random()
                norm += self.zw[i][j]
            for j in xrange(self.words):
                self.zw[i][j] /= norm

        # init pdz: p(z|d)
        self.dz.resize(self.docs)
        for i in xrange(self.docs):
            self.dz[i].resize(self.topics)
            norm = 0
            for j in xrange(self.topics):
                self.dz[i][j] = random.random()
                norm += self.dz[i][j]
            for j in xrange(self.topics):
                self.dz[i][j] /= norm

        # dw_z: p(z|d,w)
        self.dw_z.resize(self.docs)

        # init p_dw: p(d,w)
        self.p_dw.resize(self.docs)
 
    def __init__(self, corpus, topics=2):
        self._init(corpus, topics)
   
    cdef _cal_p_dw(self):
        cdef unordered_map[int, int].iterator it
        cdef float tmp
        cdef int d, z, w, wordCnt
        for d in xrange(self.docs):
            self.p_dw[d].clear()
            it = self.corpus[d].begin()
            while it != self.corpus[d].end():
                tmp = 0
                w = deref(it).first
                wordCnt = deref(it).second
                for z in xrange(self.topics):
                    tmp += (self.zw[z][w]*self.dz[d][z])**self.beta
                tmp = tmp * wordCnt
                self.p_dw[d][w] = tmp
                inc(it)

    cdef _e_step(self):
        self._cal_p_dw()
        cdef int d, w, z
        cdef vector[float] v
        cdef unordered_map[int, int].iterator it
        for d in xrange(self.docs):
            it = self.corpus[d].begin()
            while it != self.corpus[d].end():
                w = deref(it).first
                if self.dw_z[d].find(w) == self.dw_z[d].end():
                    self.dw_z[d][w] = v
                self.dw_z[d][w].clear()
                for z in xrange(self.topics):
                    self.dw_z[d][w].push_back(((self.zw[z][w] * self.dz[d][z])**self.beta) / self.p_dw[d][w])
                inc(it)

    cdef _m_step(self):
        cdef int w, z, d, idx
        cdef unordered_map[int, int].iterator it
        cdef pair[int, int] wc
        cdef float norm, tmp

        resetMatrix(self.zw, 0.0, 0)
        for z in xrange(self.topics):
            for d in xrange(self.docs):
                it = self.corpus[d].begin()
                while it != self.corpus[d].end():
                    w = deref(it).first
                    self.zw[z][w] += self.corpus[d][w] * self.dw_z[d][w][z]
                    inc(it)
            norm = 0
            for w in xrange(self.zw[z].size()):
                norm += self.zw[z][w]
            for w in xrange(self.zw[z].size()):
                self.zw[z][w] /= norm
        resetMatrix(self.dz, 0.0, 0)
        for d in xrange(self.docs):
            for z in xrange(self.topics):
                it = self.corpus[d].begin()
                while it != self.corpus[d].end():
                    w = deref(it).first
                    self.dz[d][z] += self.corpus[d][w]*self.dw_z[d][w][z]
                    inc(it)
            for z in xrange(self.topics):
                self.dz[d][z] /= self.each[d]

    cdef _cal_likelihood(self):
        self.likelihood = 0
        cdef int d, w
        cdef unordered_map[int, int].iterator it
        for d in xrange(self.docs):
            it = self.corpus[d].begin()
            while it != self.corpus[d].end():
                w = deref(it).first
                self.likelihood += self.corpus[d][w]*math.log(self.p_dw[d][w])
                inc(it)

    def train(self, max_iter=20):
        begin = time.time()
        cur = 0
        cdef int  i = 0
        for i in xrange(max_iter):
            print '%d iter' % i
            self._e_step()
            self._m_step()
            self._cal_likelihood()
            print 'likelihood %f ' % self.likelihood
            if cur != 0 and abs((self.likelihood-cur)/cur) < 1e-8:
                break
            cur = self.likelihood
        print "spend: %f seconds" % (time.time() - begin)
    def getPzw(self):
        cdef int d, z
        pzw = np.zeros([self.topics, self.words], dtype=np.float)
        for z in xrange(self.topics):
            for w in xrange(self.words):
                pzw[z, w] = self.zw[z][w]
        return pzw

    cpdef getTopics(self):
        return self.topics

    def inference(self, doc, max_iter=100):
        doc = dict(filter(lambda x:x[0]<self.words, doc.items()))
        words = sum(doc.values())
        ret = []
        for i in xrange(self.topics):
            ret.append(random.random())
        norm = sum(ret)
        for i in xrange(self.topics):
            ret[i] /= norm
        tmp = 0
        for _ in xrange(max_iter):
            p_dw = {}
            for w in doc:
                p_dw[w] = 0
                for _ in range(doc[w]):
                    for z in xrange(self.topics):
                        p_dw[w] += (ret[z]*self.zw[z][w])**self.beta
            # e setp
            dw_z = {}
            for w in doc:
                dw_z[w] = []
                for z in xrange(self.topics):
                    dw_z[w].append(((self.zw[z][w]*ret[z])**self.beta)/p_dw[w])
            # m step
            ret = [0]*self.topics
            for z in xrange(self.topics):
                for w in doc:
                    ret[z] += doc[w]*dw_z[w][z]
            for z in xrange(self.topics):
                ret[z] /= words
            # cal likelihood
            likelihood = 0
            for w in doc:
                likelihood += doc[w]*math.log(p_dw[w])
            if tmp != 0 and abs((likelihood-tmp)/tmp) < 1e-8:
                break
            tmp = likelihood
        return ret

    def post_prob_sim(self, docd, q):
        sim = 0
        for w in docd:
            tmp = 0
            for z in xrange(self.topics):
                tmp += self.zw[z][w]*q[z]
            sim += docd[w]*math.log(tmp)
        return sim

