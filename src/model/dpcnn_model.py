from keras.models import *
from keras.layers import *
from model.model_basic import BasicDeepModel
from keras.utils.vis_utils import plot_model
from keras import regularizers


dp = 7
filter_nr = 64
filter_size = 3
max_pool_size = 3
max_pool_strides = 2
dense_nr = 256
spatial_dropout = 0.2
dense_dropout = 0.5
conv_kern_reg = regularizers.l2(0.00001)
conv_bias_reg = regularizers.l2(0.00001)


class DpcnnModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'dpcnn' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):
        char_embedding = Embedding(self.max_c_features, self.char_embed_size, weights=[self.char_embedding], trainable=True, name='char_embedding')
        word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=True, name='word_embedding')

        char_input = Input(shape=(self.char_max_len,), name='char')
        word_input = Input(shape=(self.word_max_len,), name='word')
        if not self.config.main_feature == 'char':
            char_input, word_input = word_input, char_input
            char_embedding, word_embedding = word_embedding, char_embedding
            self.char_max_len, self.word_max_len = self.word_max_len, self.char_max_len

        x = char_embedding(char_input)
        x = BatchNormalization()(x)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)

        # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
        # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
        resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
        resize_emb = PReLU()(resize_emb)

        block1_output = add([block1, resize_emb])
        x = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        for i in range(dp):
            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)
            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)

            block_output = add([block1, x])
            print(i)
            if i + 1 != dp:
                x = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block_output)

        x = GlobalMaxPooling1D()(block_output)
        output = Dense(dense_nr, activation='linear')(x)
        output = BatchNormalization()(output)
        output = PReLU()(output)

        if self.config.main_feature == 'all':
            recurrent_units = 60
            word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=False, name='word_embedding')
            word_input = Input(shape=(self.word_max_len,), name='word')
            word_embedding_layer = word_embedding(word_input)
            word_embedding_layer = SpatialDropout1D(0.5)(word_embedding_layer)

            word_rnn_1 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_embedding_layer)
            word_rnn_1 = SpatialDropout1D(0.5)(word_rnn_1)
            word_rnn_2 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_rnn_1)
            word_maxpool = GlobalMaxPooling1D()(word_rnn_2)
            word_average = GlobalAveragePooling1D()(word_rnn_2)

            output = concatenate([output, word_maxpool, word_average], axis=-1)

            output = Dropout(dense_dropout)(output)
            dense2 = Dense(self.n_class, activation="softmax")(output)
            res_model = Model(inputs=[char_input, word_input], outputs=dense2)
        else:
            output = Dropout(dense_dropout)(output)
            # dense2 = Dense(self.n_class, activation="softmax", kernel_regularizer=regularizers.l2(self.wd))(output)
            dense2 = Dense(self.n_class, activation="softmax")(output)
            res_model = Model(inputs=[char_input], outputs=dense2)

        plot_model(res_model, to_file="{}.png".format(self.name), show_shapes=True)
        return res_model
