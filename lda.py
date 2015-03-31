import numpy as np
 
from gensim import matutils
from gensim.models.ldamodel import LdaModel
from gensim.models import LsiModel
from sklearn import linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
 
 
def print_features(clf, vocab, lables, n=10):
    """ Print sorted list of non-zero features/weights. """
    for i in xrange(clf.coef_.shape[0]):
        coef = clf.coef_[i]
        print '[%s] positive features: %s' % (str(lables[i]), ' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[::-1][:n] if coef[j] > 0]))
        print '[%s] negative features: %s' % (str(lables[i]), ' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[:n] if coef[j] < 0]))
 
 
def fit_classifier(X, y, C=0.1):
    """ Fit L1 Logistic Regression classifier. """
    # Smaller C means fewer features selected.
    clf = linear_model.LogisticRegression(penalty='l1', C=C)
    clf.fit(X, y)
    return clf
 
 
def fit_lda(X, vocab, num_topics=5, passes=20):
    """ Fit LDA from a scipy CSR matrix (X). """
    print 'fitting lda...'
    return LdaModel(matutils.Sparse2Corpus(X), num_topics=num_topics,
                    passes=passes,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]))

def fit_lsa(X, vocab, num_topics=5, passes=20) :
    print "fitting lsi..."
    return 
 
def print_topics(lda, vocab, n=10):
    """ Print the top words for each topic. """
    #topics = lda.show_topics(topics=-1, num_words=n, formatted=False)
    topics = lda.show_topics()
    print topics
    for ti, topic in enumerate(topics):
        print 'topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in topic))
 
 
if (__name__ == '__main__'):
    # Load data.
    rand = np.random.mtrand.RandomState(8675309)
    cats = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'talk.politics.guns', 'rec.sport.baseball']
    cats = ['comp.os.ms-windows.misc', 'rec.sport.baseball']
    data = fetch_20newsgroups(subset='train',
                              categories=cats,
                              shuffle=True,
                              random_state=rand)
    vec = CountVectorizer(min_df=10, stop_words=ENGLISH_STOP_WORDS)
    X = vec.fit_transform(data.data)
    vocab = vec.get_feature_names()
 
    # Fit classifier.
    clf = fit_classifier(X, data.target)
    print_features(clf, vocab, clf.classes_)
 
    # Fit LDA.
    lda = fit_lda(X, vocab)
    print_topics(lda, vocab)
