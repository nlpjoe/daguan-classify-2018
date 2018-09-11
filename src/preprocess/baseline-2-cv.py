import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
import os
import pickle
import time
import sys
sys.path.append('..')
from config import Config

t1=time.time()

def cal_acc(pred, label):
    right = 0
    total = 0
    for idx, p in enumerate(pred):
        total += 1
        flag = np.argmax(p)
        if int(flag) == int(label[idx]):
            right += 1
    return right / total

def cal_f_alpha(pred, label, alpha=1.0, n_out=45, verbose=False):
    # pred:  (x, 45)
    # label: (x, 1)
    # matrix = np.zeros((n_out, n_out))
    matrix = np.diag(np.array([1e-7] * n_out))
    for idx, p in enumerate(pred):
        true_label = int(label[idx])
        p = int(np.argmax(p))
        if p == true_label:
            matrix[p][p] += 1
        else:
            matrix[true_label][p] += 1

    pi = []
    ri = []
    for i in range(n_out):
        pi.append(matrix[i][i] / sum(matrix[:, i]) / n_out)
        ri.append(matrix[i][i] / sum(matrix[i, :]) / n_out)

    p = sum(pi)
    r = sum(ri)
    f = (alpha**2 + 1) * p * r / (alpha ** 2 * p + r)
    if verbose:
        # check every categories' prediction and recall
        pass
    return f


config = Config()
n_folds = 5
n_class = 19
kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)
column = "word_seg"
# column = "article"
train = pd.read_csv('../'+config.TRAIN_X)
test = pd.read_csv('../'+config.TEST_X)
test_id = test["id"].copy()
oof_predict = np.zeros((len(train[column]), n_class))
predict = np.zeros((len(test[column]), n_class))
cur_kfold = 0
accs = []
f1s = []
wins = 3
for train_index, test_index in kf.split(train[column]):
    vec = TfidfVectorizer(ngram_range=(1,wins), min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
    kfold_x_train = train[column][train_index]
    kfold_x_valid = train[column][test_index]
    k_y_train = (train['class']-1).astype(int)[train_index]
    k_y_valid = (train['class']-1).astype(int)[test_index]

    print('获得tfidf特征')
    k_trn_term_doc = vec.fit_transform(kfold_x_train)
    k_test_term_doc = vec.transform(kfold_x_valid)
    test_term_doc = vec.transform(test[column])

    # 拟合数据
    print('拟合数据')
    lin_clf = svm.LinearSVC()
    lin_clf = CalibratedClassifierCV(lin_clf)
    lin_clf.fit(k_trn_term_doc, k_y_train)

    # 预测结果
    print('预测结果')
    oof_predict[test_index] = lin_clf.predict_proba(k_test_term_doc)
    predict += lin_clf.predict_proba(test_term_doc) / n_folds

    # 计算准确度
    accuracy = cal_acc(oof_predict[test_index], k_y_valid.values)
    f1 = cal_f_alpha(oof_predict[test_index], k_y_valid.values, n_out=n_class)
    print('Test acc = %f\n' % accuracy)
    print('Test f1 = %f\n' % f1)
    accs.append(accuracy)
    f1s.append(f1)
    cur_kfold += 1

print('total acc scores is ', np.mean(accs))
print('total f1 scores is ', np.mean(f1s))

os.makedirs('./result', exist_ok=True)
d_type = 'word' if column == 'word_seg' else 'char'
with open('./result/svm{}_{}gram_oof_f1_{}_a{}.pkl'.format(d_type, wins, str(np.mean(f1s)), str(np.mean(accs))), 'wb') as f:
    pickle.dump(oof_predict, f)

with open('./result/svm{}_{}gram_pre_f1_{}_a{}.pkl'.format(d_type, wins, str(np.mean(f1s)), str(np.mean(accs))), 'wb') as f:
    pickle.dump(predict, f)

t2=time.time()
print("time use:",t2-t1)
