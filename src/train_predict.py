import os
import pandas as pd
import pickle
from config import Config
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from gensim.models.word2vec import Word2Vec

from keras.preprocessing import sequence
from keras.utils import np_utils


def static_data_prepare():
    train_y = pd.read_csv(config.TRAIN_X, usecols=['label_c_numeric']).values
    kw_train_df = pd.read_csv('../data/feature/key_words_train_feature.df')
    kw_test_df = pd.read_csv('../data/feature/key_words_test_feature.df')
    id_train_feature = kw_train_df.iloc[:, 1]
    id_test_feature = kw_test_df.iloc[:, 1]
    kw_train_feature = kw_train_df.iloc[:, 2:]
    kw_test_feature = kw_test_df.iloc[:, 2:]

    tfidf_train_feature = pd.DataFrame(pickle.load(open('../data/feature/train_x_100.pkl', 'rb')))
    tfidf_test_feature = pd.DataFrame(pickle.load(open('../data/feature/test_x_100.pkl', 'rb')))

    all_features = pd.concat((kw_train_feature, kw_test_feature), axis=0)
    scaler = MinMaxScaler()
    scaler.fit(all_features)

    kw_train_feature = pd.DataFrame(scaler.transform(kw_train_feature))
    kw_test_feature = pd.DataFrame(scaler.transform(kw_test_feature))

    train_x = pd.concat((id_train_feature, kw_train_feature, tfidf_train_feature), axis=1)
    test_x = pd.concat((id_test_feature, kw_test_feature, tfidf_test_feature), axis=1)
    train_y = np.array(train_y).reshape(-1)
    assert train_x.shape[0] == train_y.shape[0]
    print('训练数据维度', train_x.shape)
    print('训练标签维度', train_y.shape)
    print('测试数据维度', test_x.shape)

    return train_x.values, train_y, test_x.values


def deep_data_prepare(config, article_word2vec_model, word_seg_word2vec_model):
    print('深度学习模型数据准备')
    print('加载训练数据与测试数据')
    train_df = pd.read_csv(config.TRAIN_X)
    test_df = pd.read_csv(config.TEST_X)

    char_sw_list = pickle.load(open('../data/char_stopword.pkl', 'rb'))
    word_sw_list = pickle.load(open('../data/word_stopword.pkl', 'rb'))
    chi_w_list = pickle.load(open(config.chi_word_file, 'rb'))
    chi_c_list = pickle.load(open(config.chi_char_file, 'rb'))
    # 用词向量
    # 用字向量
    train_char = train_df['article']
    train_word = train_df['word_seg']
    test_char = test_df['article']
    test_word = test_df['word_seg']

    train_y = train_df['c_numerical'].values
    UNK_CHAR = len(article_word2vec_model.stoi) + 1
    PAD_CHAR = len(article_word2vec_model.stoi)

    UNK_WORD = len(word_seg_word2vec_model.stoi) + 1
    PAD_WORD = len(word_seg_word2vec_model.stoi)

    def word2id(train_dialogs, type='char', filter=True):
        if type == 'char':
            word2vec_model = article_word2vec_model
            max_len = config.CHAR_MAXLEN
            UNK = UNK_CHAR
            sw_list = set(char_sw_list)
            chi_lst = set(chi_c_list)
        elif type == 'word':
            word2vec_model = word_seg_word2vec_model
            max_len = config.WORD_MAXLEN
            UNK = UNK_WORD
            sw_list = set(word_sw_list)
            chi_lst = set(chi_w_list)
        else:
            exit('类型错误')

        print('卡方词表长：', len(chi_lst))
        train_x = []
        for d in tqdm(train_dialogs):
            d = d.split()
            line = []
            for token in d:
                if token in sw_list\
                        or token == ''\
                        or token == ' ':
                        # or token not in chi_lst\
                    continue
                if token in word2vec_model.stoi:
                    line.append(word2vec_model.stoi[token])
                else:
                    line.append(UNK)

            train_x.append(line[:max_len])
            # if type == 'word':
                # train_x.append(line[:max_len])
            # else:
                # train_x.append(line[-max_len:])
        return train_x

    train_x_word = word2id(train_word, type='word')
    train_x_char = word2id(train_char, type='char')
    test_x_char = word2id(test_char, type='char')
    test_x_word = word2id(test_word, type='word')

    train_word_left = [[UNK_WORD] + w[:-1] for w in train_x_word]
    train_word_right = [w[1:] + [UNK_WORD] for w in train_x_word]
    train_char_left = [[UNK_CHAR] + w[:-1] for w in train_x_char]
    train_char_right = [w[1:] + [UNK_CHAR] for w in train_x_char]

    test_word_left = [[UNK_WORD] + w[:-1] for w in test_x_word]
    test_word_right = [w[1:] + [UNK_WORD] for w in test_x_word]
    test_char_left = [[UNK_CHAR] + w[:-1] for w in test_x_char]
    test_char_right = [w[1:] + [UNK_CHAR] for w in test_x_char]

    UNK_CHAR = PAD_CHAR
    UNK_WORD = PAD_WORD
    train_x_char = sequence.pad_sequences(train_x_char, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    test_x_char = sequence.pad_sequences(test_x_char, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    train_x_word = sequence.pad_sequences(train_x_word, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)
    test_x_word = sequence.pad_sequences(test_x_word, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)

    train_x_char_left = sequence.pad_sequences(train_char_left, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    test_x_char_left = sequence.pad_sequences(test_char_left, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    train_x_word_left = sequence.pad_sequences(train_word_left, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)
    test_x_word_left = sequence.pad_sequences(test_word_left, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)

    train_x_char_right = sequence.pad_sequences(train_char_right, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    test_x_char_right = sequence.pad_sequences(test_char_right, maxlen=config.CHAR_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_CHAR)
    train_x_word_right = sequence.pad_sequences(train_word_right, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)
    test_x_word_right = sequence.pad_sequences(test_word_right, maxlen=config.WORD_MAXLEN, dtype='int32', padding='post', truncating='post', value=UNK_WORD)

    train_y = np_utils.to_categorical(train_y, num_classes=config.n_class)
    print('train_x char shape is: ', train_x_char.shape)
    print('test_x char shape is: ', test_x_char.shape)
    print('train_x word shape is: ', train_x_word.shape)
    print('test_x word shape is: ', test_x_word.shape)

    print('train_y shape is: ', train_y.shape)
    train = {}
    train['word'] = train_x_word
    train['char'] = train_x_char
    train['word_left'] = train_x_word_left
    train['word_right'] = train_x_word_right
    train['char_left'] = train_x_char_left
    train['char_right'] = train_x_char_right
    test = {}
    test['word'] = test_x_word
    test['char'] = test_x_char
    test['word_left'] = test_x_word_left
    test['word_right'] = test_x_word_right
    test['char_left'] = test_x_char_left
    test['char_right'] = test_x_char_right
    assert train['word_left'].shape == train['word_right'].shape == train['word'].shape
    assert train['char_left'].shape == train['char_right'].shape == train['char'].shape
    assert test['word_left'].shape == test['word_right'].shape == test['word'].shape
    assert test['char_left'].shape == test['char_right'].shape == test['char'].shape
    return train, train_y, test


def init_embedding(config, word2vec_model):
    vocab_len = len(word2vec_model.stoi) + 2
    print('Vocabulaty size : ', vocab_len)
    print('create embedding matrix')
    all_embs = np.stack(word2vec_model.embedding.values())
    embed_matrix = np.random.normal(all_embs.mean(), all_embs.std(), size=(vocab_len, config.EMBED_SIZE))
    embed_matrix[-2] = 0  # padding

    for i, val in tqdm(word2vec_model.embedding.items()):
        embed_matrix[i] = val
    return embed_matrix


def static_mode_process():
    train_x, train_y, test_x = static_data_prepare()
    model = config.model['lightgbm']()
    model.train_predict(train_x, train_y, test_x)
    # model.rerun(test_x)


def deep_data_cache():
    article_word2vec_model = Word2Vec.load(config.article_w2v_file)
    word_seg_word2vec_model = Word2Vec.load(config.word_seg_w2v_file)
    train, train_y, test = deep_data_prepare(config, article_word2vec_model, word_seg_word2vec_model)
    char_init_embed = init_embedding(config, article_word2vec_model)
    word_init_embed = init_embedding(config, word_seg_word2vec_model)
    os.makedirs('../data/cache/', exist_ok=True)
    pickle.dump((train,  train_y, test, char_init_embed, word_init_embed), open('../data/cache/clean_deep_data_{}_{}.pkl'.format(config.WORD_MAXLEN, config.EMBED_SIZE), 'wb'))


def deep_data_process():
    # deep_data_cache()
    train, train_y, test, char_init_embed, word_init_embed = pickle.load(open('../data/cache/clean_deep_data_{}_{}.pkl'.format(config.WORD_MAXLEN, config.EMBED_SIZE), 'rb'))
    config.max_c_features = len(char_init_embed)
    config.max_w_features = len(word_init_embed)
    config.char_init_embed = char_init_embed
    config.word_init_embed = word_init_embed

    model = config.model[args.model](config=config, num_flods=5)
    model.train_predict(train, train_y, test, option=config.option)
    # model.rerun(test_x)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--option', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--feature', default='word', type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=tf_config))

    config = Config()
    config.option = args.option
    config.n_gpus = len(args.gpu.split(','))
    config.main_feature = args.feature
    config.model_name = args.model
    # static_mode_process()
    deep_data_process()

