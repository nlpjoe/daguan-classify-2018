from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from keras import optimizers
import numpy as np
import pandas as pd
import os
import pickle

from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from model.snapshot import SnapshotCallbackBuilder

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from keras.utils.training_utils import multi_gpu_model
NUM_EPOCHS = 16


class BasicModel(object):

    """Docstring for BasicModel. """

    def __init__(self):
        """TODO: to be defined1. """
        pass

    def create_model(self, kfold_X_train, y_train, kfold_X_test, y_test, test):
        pass

    def cal_acc(self, pred, label):
        right = 0
        total = 0
        for idx, p in enumerate(pred):
            total += 1
            flag = np.argmax(p)
            if int(flag) == int(label[idx]):
                right += 1
        return right / total

    def cal_f_alpha(self, pred, label, alpha=1.0, n_out=45, verbose=False):
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


class BasicDeepModel(BasicModel):

    """Docstring for BasicModel. """

    def __init__(self, n_folds=5, name='BasicModel', config=None):
        if config is None:
            exit('请传入数值')
        self.name = name
        self.config = config
        self.n_class = config.n_class
        # char 特征
        self.char_max_len = config.CHAR_MAXLEN
        self.max_c_features = config.max_c_features
        # word 特征
        self.word_max_len = config.WORD_MAXLEN
        self.max_w_features = config.max_w_features
        self.char_mask_value = self.max_c_features - 2
        self.word_mask_value = self.max_w_features - 2
        self.batch_size = config.BATCH_SIZE

        self.char_embedding = config.char_init_embed
        self.word_embedding = config.word_init_embed
        self.char_embed_size = len(self.char_embedding[0])
        self.word_embed_size = len(self.word_embedding[0])
        self.n_folds = n_folds

        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)
        M = 3  # number of snapshots
        # alpha_zero = 5e-4  # initial learning rate
        # self.snap_epoch = NUM_EPOCHS
        # self.snapshot = SnapshotCallbackBuilder(self.snap_epoch, M, alpha_zero)
        self.last_val_acc = 0.

        self.init_lr = 0.001
        self.lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.000001, verbose=1)

        #  if self.config.option == 6:
            #  self.init_lr = 1e-3
        #  elif self.config.option == 5:
            #  if 'attention' in self.config.model_name:
                #  self.wd = 0.001
            #  if 'textcnn' in self.config.model_name:
                #  self.init_lr = 0.001
                #  self.wd = 0.0015
            #  if 'capsule' in self.config.model_name:
                #  self.init_lr = 0.001
                #  self.wd = 0.003
            #  if 'lstmgru' in self.config.model_name:
                #  self.init_lr = 0.001
        #  elif self.config.option == 4:
            #  self.init_lr = 0.001
        #  elif self.config.option == 3:
            #  self.init_lr = 0.002
            #  # self.poly_decay = self.poly_decay_attention
        #  else:
            #  self.init_lr = 1e-3
        self.snapshot = SnapshotCallbackBuilder(NUM_EPOCHS, M, self.init_lr)
        self.early_stop_monitor = EarlyStopping(patience=5)
        print("[INFO] training with {} GPUs...".format(config.n_gpus))

        self.wd = config.wd
        self.model = self.create_model()
        if config.n_gpus > 1:
            self.model = multi_gpu_model(self.model, gpus=config.n_gpus)

    def poly_decay_attention(self, epoch):
        # initialize the maximum number of epochs, base learning rate,
        # and power of the polynomial

        if epoch < 5:
            print('epoch:{}, lr:{}, wd:{}'.format(1+epoch, self.init_lr, self.wd))
            return self.init_lr
        maxEpochs = NUM_EPOCHS
        baseLR = self.init_lr
        power = 1.0

        # compute the new learning rate based on polynomial decay
        alpha = baseLR * (1 - (epoch / (float(maxEpochs)))) ** power

        print('epoch:{}, lr:{}, wd:{}'.format(1+epoch, alpha, self.wd))

        # return the new learning rate
        return alpha

    def poly_decay(self, epoch):
        initial_lrate = self.init_lr
        drop = 0.5
        epochs_drop = 12
        lrate = initial_lrate * (drop ** ((1+epoch)//epochs_drop))
        print('epoch:{}, lr:{}, wd:{}'.format(1+epoch, lrate, self.wd))
        return lrate

    def plot_loss(self, H, fold):
        # grab the history object dictionary
        H = H.history

        # plot the training loss and accuracy
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H["loss"], label="train_loss")
        plt.plot(N, H["val_loss"], label="test_loss")
        plt.plot(N, H["acc"], label="train_acc")
        plt.plot(N, H["val_acc"], label="test_acc")
        plt.title("model {} option {}".format(self.name, self.config.option))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        # save the figure
        os.makedirs('loss', exist_ok=True)
        plt.savefig('loss/{}-op{}-fold{}.png'.format(self.name, self.config.option, fold))
        plt.close()

    def plot_loss_option3(self, H1, H2, fold):
        # grab the history object dictionary
        H1 = H1.history
        H2 = H2.history
        H = {}
        H['loss'] = H1['loss'] + H2['loss']
        H['val_loss'] = H1['val_loss'] + H2['val_loss']
        H['acc'] = H1['acc'] + H2['acc']
        H['val_acc'] = H1['val_acc'] + H2['val_acc']
        # plot the training loss and accuracy
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H["loss"], label="train_loss")
        plt.plot(N, H["val_loss"], label="test_loss")
        plt.plot(N, H["acc"], label="train_acc")
        plt.plot(N, H["val_acc"], label="test_acc")
        plt.title("model {} option {}".format(self.name, self.config.option))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        # save the figure
        os.makedirs('loss', exist_ok=True)
        plt.savefig('loss/{}-op{}-fold{}.png'.format(self.name, self.config.option, fold))
        plt.close()


    def train_predict(self, train, train_y, test, option=3):
        """
        we use KFold way to train our model and save the model
        :param train:
        :return:
        """
        name = self.name
        model_name = '../ckpt-op{}/{}'.format(self.config.option, self.name)
        os.makedirs(model_name, exist_ok=True)

        self.model.save_weights(model_name + '/init_weight.h5')

        count_kflod = 0
        predict = np.zeros((len(test['word']), self.n_class))
        oof_predict = np.zeros((len(train['word']), self.n_class))
        scores_acc = []
        scores_f1 = []
        for train_index, test_index in self.kf.split(train['word']):
            kfold_X_train = {}
            kfold_X_valid = {}
            model_prefix = model_name + '/' + str(count_kflod)
            if not os.path.exists(model_prefix):
                os.mkdir(model_prefix)
            filepath = model_prefix + '/' + str(count_kflod) + 'model.h5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            y_train, y_test = train_y[train_index], train_y[test_index]

            self.model.load_weights(model_name + '/init_weight.h5')

            for c in ['word', 'char', 'word_left', 'word_right', 'char_left', 'char_right']:
                kfold_X_train[c] = train[c][train_index]
                kfold_X_valid[c] = train[c][test_index]

            if option == 1:
                # 冻结embedding， 并且使用snapshot的方式来训练模型
                adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=2.0)
                self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
                self.model.summary()
                self.model.fit(kfold_X_train, y_train,
                               batch_size=self.batch_size * self.config.n_gpus,
                               epochs=self.snap_epoch,
                               verbose=1,
                               validation_data=(kfold_X_valid, y_test),
                               callbacks=self.snapshot.get_callbacks(model_save_place=model_prefix))

            elif option == 2:
                # 前期冻结embedding层，训练好参数后，开放enbedding层并且使用snapshot的方式来训练模型
                adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=2)
                self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
                self.model.summary()
                H = self.model.fit(kfold_X_train, y_train,
                               batch_size=self.batch_size * self.config.n_gpus,
                               epochs=6,
                               verbose=1,
                               validation_data=(kfold_X_valid, y_test))

                if self.config.main_feature == 'all':
                    self.model.get_layer('char_embedding').trainable = True
                    self.model.get_layer('word_embedding').trainable = True
                elif self.config.main_feature == 'word':
                    self.model.get_layer('word_embedding').trainable = True
                elif self.config.main_feature == 'char':
                    self.model.get_layer('char_embedding').trainable = True
                else:
                    exit('Wrong feature')
                self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                H = self.model.fit(kfold_X_train, y_train,
                               batch_size=self.batch_size * self.config.n_gpus,
                               epochs=self.snap_epoch,
                               verbose=1,
                               validation_data=(kfold_X_valid, y_test),
                               callbacks=self.snapshot.get_callbacks(model_save_place=model_prefix))

            elif option == 3:

                # 前期冻结embedding层，训练好参数后，开放enbedding层继续训练模型
                if self.config.main_feature == 'all':
                    self.model.get_layer('char_embedding').trainable = False
                    self.model.get_layer('word_embedding').trainable = False
                elif self.config.main_feature == 'word':
                    self.model.get_layer('word_embedding').trainable = False
                elif self.config.main_feature == 'char':
                    self.model.get_layer('char_embedding').trainable = False
                else:
                    exit('Wrong feature')

                # callbacks = [LearningRateScheduler(self.poly_decay)]
                adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=2.4)
                self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
                self.model.summary()
                H1 = self.model.fit(kfold_X_train, y_train,
                                    batch_size=self.batch_size * self.config.n_gpus,
                                    epochs=2,
                                    verbose=1,
                                    validation_data=(kfold_X_valid, y_test))

                if self.config.main_feature == 'all':
                    self.model.get_layer('char_embedding').trainable = True
                    self.model.get_layer('word_embedding').trainable = True
                elif self.config.main_feature == 'word':
                    self.model.get_layer('word_embedding').trainable = True
                elif self.config.main_feature == 'char':
                    self.model.get_layer('char_embedding').trainable = True
                else:
                    exit('Wrong feature')
                print('放开embedding训练')

                callbacks = [
                    self.lr_schedule,
                    checkpoint,
                ]
                adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=1.5)
                self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
                self.model.summary()

                H2 = self.model.fit(kfold_X_train, y_train,
                                    batch_size=self.batch_size * self.config.n_gpus,
                                    epochs=10,
                                    verbose=1,
                                    validation_data=(kfold_X_valid, y_test),
                                    callbacks=callbacks)

                # self.model.save_weights(model_prefix + '/' + str(count_kflod) + 'model.h5')
                self.plot_loss_option3(H1, H2, count_kflod)

            elif option == 4:

                if self.config.n_gpus == 1:
                    if self.config.main_feature == 'all':
                        self.model.get_layer('char_embedding').trainable = True
                        self.model.get_layer('word_embedding').trainable = True
                    elif self.config.main_feature == 'word':
                        self.model.get_layer('word_embedding').trainable = True
                    elif self.config.main_feature == 'char':
                        self.model.get_layer('char_embedding').trainable = True
                    else:
                        exit('Wrong feature')
                opt = optimizers.SGD(lr=self.init_lr, momentum=0.9, decay=1e-6)
                self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
                self.model.summary()
                callbacks = [
                    LearningRateScheduler(self.poly_decay),
                    self.early_stop_monitor,
                ]

                H = self.model.fit(kfold_X_train, y_train,
                               batch_size=self.batch_size * self.config.n_gpus,
                               epochs=NUM_EPOCHS,
                               verbose=1,
                               validation_data=(kfold_X_valid, y_test),
                               callbacks=callbacks)
                self.plot_loss(H, count_kflod)
                self.model.save_weights(model_prefix + '/' + str(count_kflod) + 'model.h5')

            elif option == 5:
                # adam 目前最佳

                # if self.config.n_gpus == 1:
                    # if self.config.main_feature == 'all':
                        # self.model.get_layer('char_embedding').trainable = True
                        # self.model.get_layer('word_embedding').trainable = True
                    # elif self.config.main_feature == 'word':
                        # self.model.get_layer('word_embedding').trainable = True
                    # elif self.config.main_feature == 'char':
                        # self.model.get_layer('char_embedding').trainable = True
                    # else:
                        # exit('Wrong feature')

                #  if self.config.model_name == 'rnn_attention':
                    #  opt = optimizers.SGD(lr=0.2, decay=1e-6, momentum=0.95, nesterov=True)
                opt = optimizers.Adam(lr=1e-3, clipnorm=1.0)
                #  opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
                self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
                #  self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
                self.model.summary()
                callbacks = [
                    checkpoint,
                    self.lr_schedule,
                ]

                H = self.model.fit(kfold_X_train, y_train,
                                batch_size=self.batch_size * self.config.n_gpus,
                                epochs=20,
                                verbose=1,
                                validation_data=(kfold_X_valid, y_test),
                                callbacks=callbacks)
                self.plot_loss(H, count_kflod)
                #  self.model.save_weights(model_prefix + '/' + str(count_kflod) + 'model.h5')

            elif option == 6:
                # snapshot + adam

                if self.config.n_gpus == 1:
                    if self.config.main_feature == 'all':
                        self.model.get_layer('char_embedding').trainable = True
                        self.model.get_layer('word_embedding').trainable = True
                    elif self.config.main_feature == 'word':
                        self.model.get_layer('word_embedding').trainable = True
                    elif self.config.main_feature == 'char':
                        self.model.get_layer('char_embedding').trainable = True
                    else:
                        exit('Wrong feature')
                opt = optimizers.Adam(lr=self.init_lr, decay=1e-6)
                self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
                self.model.summary()
                H = self.model.fit(kfold_X_train, y_train,
                               batch_size=self.batch_size * self.config.n_gpus,
                               epochs=NUM_EPOCHS,
                               verbose=1,
                               validation_data=(kfold_X_valid, y_test),
                               callbacks=callbacks)

                self.plot_loss(H, count_kflod)
                #  self.model.save_weights(model_prefix + '/' + str(count_kflod) + 'model.h5')

            else:
                exit('Wrong option')

            evaluations = []
            for i in os.listdir(model_prefix):
                if '.h5' in i:
                    evaluations.append(i)
            print(evaluations)

            preds1 = np.zeros((test['word'].shape[0], self.n_class))
            preds2 = np.zeros((len(kfold_X_valid['word']), self.n_class))
            for run, i in enumerate(evaluations):
                self.model.load_weights(os.path.join(model_prefix, i))
                preds1 += self.model.predict(test, verbose=1) / len(evaluations)
                preds2 += self.model.predict(kfold_X_valid, batch_size=64*self.config.n_gpus) / len(evaluations)

                # model.save_weights('./ckpt/DNN_SNAP/' + str(count_kflod) + 'DNN.h5')

            # results = model.predict(test, verbose=1)

            predict += preds1 / self.n_folds
            oof_predict[test_index] = preds2

            accuracy = self.cal_acc(oof_predict[test_index], np.argmax(y_test, axis=1))
            f1 = self.cal_f_alpha(oof_predict[test_index], np.argmax(y_test, axis=1), n_out=self.n_class)

            print('the kflod cv acc is : ', str(accuracy))
            print('the kflod cv f1 is : ', str(f1))
            count_kflod += 1
            scores_acc.append(accuracy)
            scores_f1.append(f1)

        print('total acc scores is ', np.mean(scores_acc))
        print('total f1 scores is ', np.mean(scores_f1))

        os.makedirs('../data/result-op{}'.format(self.config.option), exist_ok=True)
        with open('../data/result-op{}/{}_oof_f1_{}_a{}.pkl'.format(self.config.option, name, str(np.mean(scores_f1)), str(np.mean(scores_acc))), 'wb') as f:
            pickle.dump(oof_predict, f)

        with open('../data/result-op{}/{}_pre_f1_{}_a{}.pkl'.format(self.config.option, name, str(np.mean(scores_f1)), str(np.mean(scores_acc))), 'wb') as f:
            pickle.dump(predict, f)

        print('done')

    def rerun(self, test):
        name = self.name
        evaluations = []
        for i in range(4):
            evaluations.append('../ckpt/{}/{}/{}model.h5'.format(name, i, i))

        predict = np.zeros((len(test), self.n_class))
        preds1 = np.zeros((test.shape[0], self.n_class))

        for run, i in enumerate(evaluations):
            self.model.load_weights(i)
            preds1 += self.model.predict(test, verbose=1) / len(evaluations)

        predict += preds1 / 4

        with open('../data/result/' + name + '_pre_.pkl', 'wb') as f:
            pickle.dump(predict, f)

class BasicStaticModel(BasicModel):

    def __init__(self, params, n_folds=5, name='BasicStaticModel', n_class=45):
        self.params = params
        self.n_folds = n_folds
        self.name = name
        self.kf = KFold(n_splits=5, shuffle=True, random_state=10)
        self.n_class = n_class

    def tuning_model(self, X_train, y_train):
        pass

    def train_predict(self, train, train_y, test):
        name = self.name
        count_kfold = 0
        model_name = '../ckpt/' + name
        os.makedirs(model_name, exist_ok=True)

        predict = np.zeros((test.shape[0], self.n_class))
        oof_predict = np.zeros((train.shape[0], self.n_class))
        scores_acc = []
        scores_f1 = []
        train = train[:, 1:]
        test = test[:, 1:]

        for train_index, val_index in self.kf.split(train):

            model_prefix = model_name + '/' + str(count_kfold)
            os.makedirs(model_prefix, exist_ok=True)

            kfold_X_train, kfold_X_val = train[train_index], train[val_index]
            y_train, y_val = train_y[train_index], train_y[val_index]

            pred, results, best = self.create_model(kfold_X_train, y_train,
                                                    kfold_X_val, y_val,
                                                    test)
            best.save_model(model_prefix + '/' + str(count_kfold) + 'model.h5')

            score = self.cal_acc(pred, y_val)
            score_f = self.cal_f_alpha(pred, y_val, n_out=self.n_class)
            print('Test acc = %f\n' % score)
            print('Test f1 = %f\n' % score_f)
            scores_acc.append(score)
            scores_f1.append(score_f)

            predict += np.divide(results, self.n_folds)
            oof_predict[val_index] = pred
            count_kfold += 1
        print("Total acc'mean is ", np.mean(scores_acc))
        print("Total f1'mean is ", np.mean(scores_f1))

        # 保存结果
        os.makedirs('../data/result', exist_ok=True)

        with open('../data/result/{}_oof_f1_{}_a{}.pkl'.format(name, str(np.mean(scores_f1)), str(np.mean(scores_acc))), 'wb') as f:
            pickle.dump(oof_predict, f)

        with open('../data/result/{}_pre_f1_{}_a{}.pkl'.format(name, str(np.mean(scores_f1)), str(np.mean(scores_acc))), 'wb') as f:
            pickle.dump(predict, f)

        print('done')

    def rerun(self, test):
        name = self.name
        evaluations = []
        for i in range(5):
            evaluations.append('../ckpt/{}/{}/{}model.h5'.format(name, i, i))
        test = test[:, 1:]

        predict = np.zeros((len(test), self.n_class))
        preds1 = np.zeros((test.shape[0], self.n_class))

        for run, i in enumerate(evaluations):
            self.model = lgbm.Booster(model_file=i)
            preds1 += self.model.predict(test, verbose=1) / len(evaluations)

        predict += preds1 / 5

        with open('../data/result/' + name + '_pre_.pkl', 'wb') as f:
            pickle.dump(predict, f)


if __name__ == '__main__':
    bm = BasicModel()
    print(bm.cal_f_alpha([[0, 0, 1], [0, 1, 0], [0, 1, 0]], [2, 1, 0], alpha=1.0, n_out=3))

