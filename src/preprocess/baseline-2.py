import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm

column = "word_seg"
train = pd.read_csv('../../data/raw_data/train_set.csv')
test = pd.read_csv('../../data/raw_data/test_set.csv')
test_id = test["id"].copy()
vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
fid0=open('baseline.csv','w')

y=(train["class"]-1).astype(int)
lin_clf = svm.LinearSVC()
lin_clf.fit(trn_term_doc,y)
preds = lin_clf.predict(test_term_doc)
i=0
fid0.write("id,class"+"\n")
for item in preds:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()
