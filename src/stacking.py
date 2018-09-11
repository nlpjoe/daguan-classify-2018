import pickle
import glob
import pandas as pd
from config import Config
from keras.utils import np_utils
from keras.layers import *
from model.snapshot import SnapshotCallbackBuilder
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from model.model_basic import BasicModel

import os


def data_prepare():
    train_df = pd.read_csv(config.TRAIN_X)
    train_y = train_df['c_numerical'].values
    train_y = np_utils.to_categorical(train_y, num_classes=config.n_class)

    oof_filename = []
    test_filename = []

    # tf-idf features
    w_features = pickle.load(open('../data/feature/train_x_word_250.pkl', 'rb'))
    c_features = pickle.load(open('../data/feature/train_x_char_250.pkl', 'rb'))
    test_w_features = pickle.load(open('../data/feature/test_x_word_250.pkl', 'rb'))
    test_c_features = pickle.load(open('../data/feature/test_x_char_250.pkl', 'rb'))

    option = args.option
    # oof features
    if option == 0:
        filenames = glob.glob('../data/all_best/*oof*')

    if option == 1:
        filenames = glob.glob('../data/result-op1/*oof*')
    elif option == 2:
        filenames = glob.glob('../data/result-op2/*oof*')
    elif option == 3:
        filenames = glob.glob('../data/result-op3/*oof*')
    elif option == 4:
        filenames = glob.glob('../data/result-op4/*oof*')
    elif option == 5:
        filenames = glob.glob('../data/result-op5/*oof*')
    elif option == 6:
        filenames = glob.glob('../data/result-op6/*oof*')
    # elif option == 123:
        # filenames = glob.glob('../data/result-op1/*oof*') +\
                        # glob.glob('../data/result-op2/*oof*') + \
                        # glob.glob('../data/result-op3/*oof*')
    # elif option == 23:
        # filenames = glob.glob('../data/result-op2/*oof*') +\
                        # glob.glob('../data/result-op3/*oof*')
    # elif option == 13:
        # filenames = glob.glob('../data/result-op1/*oof*') +\
                        # glob.glob('../data/result-op3/*oof*')

    filenames = [e for e in filenames if 'dpcnn' not in e]
    from pprint import pprint
    pprint(filenames)
    for filename in filenames:
        oof_filename.append(filename)
        # rerun时需用
        # filename = '_'.join(filename.split('_')[:-1]) + '_.pkl'
        filename = filename.replace('_oof_', '_pre_')
        test_filename.append(filename)

    for i, (tra, tes) in enumerate(zip(oof_filename, test_filename)):
        oof_feature = pickle.load(open(tra, 'rb'))
        print(tra ,oof_feature.shape)
        oof_data = oof_feature if i == 0 else np.concatenate((oof_data, oof_feature), axis=-1)
        oof_feature = pickle.load(open(tes, 'rb'))
        print(tes,oof_feature.shape)
        test_data = oof_feature if i == 0 else np.concatenate((test_data, oof_feature), axis=-1)

    if args.tfidf:
        print(test_w_features.shape)
        print(test_c_features.shape)
        print(test_data.shape)
        train_x = np.concatenate((w_features, c_features, oof_data), axis=-1)
        test_x = np.concatenate((test_w_features, test_c_features, test_data), axis=-1)
    else:
        train_x = oof_data
        test_x = test_data
    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    print('test_x shape: ', test_x.shape)

    return train_x, train_y, test_x


def get_model(train_x):
    input_shape = Input(shape=(train_x.shape[1],), name='dialogs')
    x = Dense(256, activation='relu')(input_shape)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(config.n_class, activation="softmax")(x)
    res_model = Model(inputs=[input_shape], outputs=x)
    return res_model


# 第一次stacking
def stacking_first(train, train_y, test):
    savepath = './stack_op{}_dt{}_tfidf{}/'.format(args.option, args.data_type, args.tfidf)
    os.makedirs(savepath, exist_ok=True)

    count_kflod = 0
    num_folds = 6
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test.shape[0], config.n_class))
    oof_predict = np.zeros((train.shape[0], config.n_class))
    scores = []
    f1s = []

    for train_index, test_index in kf.split(train):

        kfold_X_train = {}
        kfold_X_valid = {}

        y_train, y_test = train_y[train_index], train_y[test_index]

        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        model_prefix = savepath + 'DNN' + str(count_kflod)
        if not os.path.exists(model_prefix):
            os.mkdir(model_prefix)

        M = 4  # number of snapshots
        alpha_zero = 1e-3  # initial learning rate
        snap_epoch = 16
        snapshot = SnapshotCallbackBuilder(snap_epoch, M, alpha_zero)

        res_model = get_model(train)
        res_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # res_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,  class_weight=class_weight)
        res_model.fit(kfold_X_train, y_train, batch_size=BATCH_SIZE, epochs=snap_epoch, verbose=1,
                      validation_data=(kfold_X_valid, y_test),
                      callbacks=snapshot.get_callbacks(model_save_place=model_prefix))

        evaluations = []
        for i in os.listdir(model_prefix):
            if '.h5' in i:
                evaluations.append(i)

        preds1 = np.zeros((test.shape[0], config.n_class))
        preds2 = np.zeros((len(kfold_X_valid), config.n_class))
        for run, i in enumerate(evaluations):
            res_model.load_weights(os.path.join(model_prefix, i))
            preds1 += res_model.predict(test, verbose=1) / len(evaluations)
            preds2 += res_model.predict(kfold_X_valid, batch_size=128) / len(evaluations)

        predict += preds1 / num_folds
        oof_predict[test_index] = preds2

        accuracy = mb.cal_acc(oof_predict[test_index], np.argmax(y_test, axis=1))
        f1 = mb.cal_f_alpha(oof_predict[test_index], np.argmax(y_test, axis=1), n_out=config.n_class)
        print('the kflod cv is : ', str(accuracy))
        print('the kflod f1 is : ', str(f1))
        count_kflod += 1
        scores.append(accuracy)
        f1s.append(f1)
    print('total scores is ', np.mean(scores))
    print('total f1 is ', np.mean(f1s))
    return predict


# 使用pseudo-labeling做第二次stacking
def stacking_pseudo(train, train_y, test, results):
    answer = np.argmax(results, axis=1)
    answer = np_utils.to_categorical(answer, num_classes=config.n_class)

    train_y = np.concatenate([train_y, answer], axis=0)
    train = np.concatenate([train, test], axis=0)

    savepath = './pesudo_{}/'.format(args.option)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    count_kflod = 0
    num_folds = 6
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test.shape[0], config.n_class))
    oof_predict = np.zeros((train.shape[0], config.n_class))
    scores = []
    f1s = []
    for train_index, test_index in kf.split(train):

        kfold_X_train = {}
        kfold_X_valid = {}

        y_train, y_test = train_y[train_index], train_y[test_index]

        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        model_prefix = savepath + 'DNN' + str(count_kflod)
        if not os.path.exists(model_prefix):
            os.mkdir(model_prefix)

        M = 4  # number of snapshots
        alpha_zero = 1e-3  # initial learning rate
        snap_epoch = 16
        snapshot = SnapshotCallbackBuilder(snap_epoch, M, alpha_zero)

        res_model = get_model(train)
        res_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # res_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,  class_weight=class_weight)
        res_model.fit(kfold_X_train, y_train, batch_size=BATCH_SIZE, epochs=snap_epoch, verbose=1,
                      validation_data=(kfold_X_valid, y_test),
                      callbacks=snapshot.get_callbacks(model_save_place=model_prefix))

        evaluations = []
        for i in os.listdir(model_prefix):
            if '.h5' in i:
                evaluations.append(i)
        print(evaluations)

        preds1 = np.zeros((test.shape[0], config.n_class))
        preds2 = np.zeros((len(kfold_X_valid), config.n_class))
        for run, i in enumerate(evaluations):
            res_model.load_weights(os.path.join(model_prefix, i))
            preds1 += res_model.predict(test, verbose=1) / len(evaluations)
            preds2 += res_model.predict(kfold_X_valid, batch_size=128) / len(evaluations)

        predict += preds1 / num_folds
        oof_predict[test_index] = preds2

        accuracy = mb.cal_acc(oof_predict[test_index], np.argmax(y_test, axis=1))
        f1 = mb.cal_f_alpha(oof_predict[test_index], np.argmax(y_test, axis=1), n_out=config.n_class)
        print('the kflod cv is : ', str(accuracy))
        print('the kflod f1 is : ', str(f1))
        count_kflod += 1
        scores.append(accuracy)
        f1s.append(f1)
    print('total scores is ', np.mean(scores))
    print('total f1 is ', np.mean(f1s))
    return predict


def save_result(predict, prefix):
    os.makedirs('../data/result', exist_ok=True)
    with open('../data/result/{}.pkl'.format(prefix), 'wb') as f:
        pickle.dump(predict, f)

    res = pd.DataFrame()
    test_id = pd.read_csv(config.TEST_X)

    label_stoi = pickle.load(open('../data/label_stoi.pkl', 'rb'))
    label_itos = {v: k for k, v in label_stoi.items()}

    results = np.argmax(predict, axis=-1)
    results = [label_itos[e] for e in results]
    res['id'] = test_id['id']
    res['class'] = results
    res.to_csv(prefix+'.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='6', type=str)
    parser.add_argument('--option', default=3, type=int)
    parser.add_argument('--data_type', default='word', type=str)
    parser.add_argument('--tfidf', type=bool)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=tf_config))

    mb = BasicModel()
    config = Config()
    BATCH_SIZE = config.BATCH_SIZE
    train, train_y, test = data_prepare()
    predicts = stacking_first(train, train_y, test)
    save_result(predicts, prefix='stacking_first_op{}_{}_{}'.format(args.option, args.data_type, args.tfidf))
    predicts = stacking_pseudo(train, train_y, test, predicts)
    save_result(predicts, prefix='stacking_pseudo_op{}'.format(args.option))

