from keras.models import *
from keras.layers import *
from model.model_basic import BasicDeepModel
from keras.utils.vis_utils import plot_model
from keras import regularizers


hidden_dim = 100

class LstmgruModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'lstmgru' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):
        dropout_p = 0.2
        if self.config.main_feature == 'word':
            dropout_p = 0.33
        char_embedding = Embedding(self.max_c_features, self.char_embed_size, weights=[self.char_embedding], trainable=True, name='char_embedding')
        word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=True, name='word_embedding')
        char_input = Input(shape=(self.char_max_len,), name='char')
        word_input = Input(shape=(self.word_max_len,), name='word')

        c_mask = Masking(mask_value=self.char_mask_value)(char_input)
        w_mask = Masking(mask_value=self.word_mask_value)(word_input)
        char_embedding = Embedding(self.max_c_features, self.char_embed_size, weights=[self.char_embedding], trainable=True, name='char_embedding')
        word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=True, name='word_embedding')

        if self.config.main_feature == 'char':
            embedding = char_embedding
            input = char_input
            x = c_mask
        else:
            embedding = word_embedding
            input = word_input
            x = w_mask

        x = embedding(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(dropout_p)(x)
        x = Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True))(x)
        x = SpatialDropout1D(dropout_p)(x)
        x = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)

        if self.config.main_feature == 'all':
            recurrent_units = 60
            word_embedding_layer = word_embedding(word_input)
            word_embedding_layer = SpatialDropout1D(0.5)(word_embedding_layer)
            word_rnn_1 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_embedding_layer)
            word_rnn_1 = SpatialDropout1D(0.1)(word_rnn_1)
            word_rnn_2 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_rnn_1)
            word_maxpool = GlobalMaxPooling1D()(word_rnn_2)
            word_average = GlobalAveragePooling1D()(word_rnn_2)
            concat2 = concatenate([avg_pool, max_pool, word_maxpool, word_average], axis=-1)
            dense2 = Dense(self.n_class, activation="softmax")(concat2)
            res_model = Model(inputs=[char_input, word_input], outputs=dense2)

        else:
            concat2 = concatenate([avg_pool, max_pool], axis=-1)
            #  all_views = BatchNormalization()(all_views)
            concat2 = Dropout(dropout_p)(concat2)
            dense2 = Dense(self.n_class, activation="softmax", kernel_regularizer=regularizers.l2(self.wd))(concat2)
            res_model = Model(inputs=[input], outputs=dense2)
        plot_model(res_model, to_file="{}.png".format(self.name), show_shapes=True)
        return res_model
