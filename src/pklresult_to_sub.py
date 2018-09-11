import config
import pandas as pd
import pickle
import numpy as np


def save_result(res_pkl, prefix):

    res = pd.DataFrame()
    predict = pickle.load(open(res_pkl, 'rb'))
    test_id = pd.read_csv(config.TEST_X)

    label_stoi = pickle.load(open('../data/label_stoi.pkl', 'rb'))
    label_itos = {v: k for k, v in label_stoi.items()}

    results = np.argmax(predict, axis=-1)
    results = [label_itos[e] for e in results]
    res['id'] = test_id['id']
    res['class'] = results
    res.to_csv(prefix+'.csv', index=False)
    #  sort_lst = sorted(count_categories.items(), key=lambda x: x[1], reverse=True)
    #  os.makedirs('../data/analysis/', exist_ok=True)
    #  pickle.dump(sort_lst, open('../data/analysis/{}-categories_num.pkl'.format(prefix), 'wb'))


if __name__ == '__main__':
    config = config.Config()
    save_result('../data/result-op5/textCNN2word_pre_f1_0.7611465619606355_a0.776127630049537.pkl', '../data/result-op5/textcnnword')
