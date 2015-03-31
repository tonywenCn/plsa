from cplsa import *

corpus = [{0:2,3:5},{0:5,2:1},{1:2,4:5}]
p = Plsa(corpus)
p.train()
z = p.inference({0:4, 6:7})
