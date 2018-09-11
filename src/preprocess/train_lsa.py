import os
import pandas as pd
import pickle
import sys
sys.path.extend(['../'])
from config import Config

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

cfg = Config()
train_df = pd.read_csv('../' + cfg.TRAIN_X)
test_df = pd.read_csv('../' + cfg.TEST_X)
try:
    sw_list = set(e.strip() for e in open('../../data/stopword.txt'))
except:
    sw_list = []
if len(sys.argv) != 3:
    exit()

feature = str(sys.argv[1])
n_dim = int(sys.argv[2])

train_X = []
test_X = []
corpus = train_df['word_seg'] if feature == 'word' else train_df['article']
corpus_test = test_df['word_seg'] if feature == 'word' else test_df['article']
for d in corpus:
    train_X.append(d)

for d in corpus_test:
    test_X.append(d)

vectorizer = TfidfVectorizer(ngram_range=(1,2),
                             stop_words=sw_list,
                             sublinear_tf=True,
                             use_idf=True,
                             norm='l2',
                             max_features=10000
                             )
# vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

svd = TruncatedSVD(n_components=n_dim)
lsa = make_pipeline(vectorizer, svd)

print('Fit data...')
train_X = lsa.fit_transform(train_X)
test_X = lsa.transform(test_X)

print('save make pipeline')
os.makedirs('../../data/feature/', exist_ok=True)
with open('../../data/feature/make_pipeline_{}_{}.pkl'.format(feature, n_dim), 'wb') as f:
    pickle.dump(lsa, f)

print('save result')
with open('../../data/feature/train_x_{}_{}.pkl'.format(feature, n_dim), 'wb') as f:
    pickle.dump(train_X, f)

with open('../../data/feature/test_x_{}_{}.pkl'.format(feature, n_dim), 'wb') as f:
    pickle.dump(test_X, f)
print('done')
