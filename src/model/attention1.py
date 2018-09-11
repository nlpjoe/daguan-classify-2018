from keras.layers import *
from keras.models import *
from model.model_component import AttentionWeightedAverage
from model.model_basic import BasicDeepModel
from keras.utils.vis_utils import plot_model

class AttentionNoLamdaModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'attention' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):

        recurrent_units = 60

        char_embedding = Embedding(self.max_c_features, self.char_embed_size, weights=[self.char_embedding], trainable=True, name='char_embedding')
        word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=True, name='word_embedding')

        char_input = Input(shape=(self.char_max_len,), name='char')
        word_input = Input(shape=(self.word_max_len,), name='word')
        if not self.config.main_feature == 'char':
            char_input, word_input = word_input, char_input
            char_embedding, word_embedding = word_embedding, char_embedding
            self.char_max_len, self.word_max_len = self.word_max_len, self.char_max_len

        x = char_embedding(char_input)
        x = SpatialDropout1D(0.25)(x)
        # x = BatchNormalization()(x)

        rnn_1 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
        rnn_1 = SpatialDropout1D(0.2)(rnn_1)
        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(rnn_1)
        # x = concatenate([rnn_1, rnn_2], axis=2)

        # last = Lambda(lambda t: t[:, -1], name='last')(x)
        maxpool = GlobalMaxPooling1D()(x)
        attn = AttentionWeightedAverage()(x)
        average = GlobalAveragePooling1D()(x)

        if self.config.main_feature == 'all':
            word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=False, name='word_embedding')
            word_input = Input(shape=(self.word_max_len,), name='word')
            word_embedding_layer = word_embedding(word_input)
            word_embedding_layer = SpatialDropout1D(0.25)(word_embedding_layer)

            word_embedding_layer = Dropout(0.1)(word_embedding_layer)
            word_rnn_1 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_embedding_layer)
            word_rnn_1 = SpatialDropout1(0.1)(word_rnn_1)
            word_rnn_2 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_rnn_1)

            word_maxpool = GlobalMaxPooling1D()(word_rnn_2)
            word_average = GlobalAveragePooling1D()(word_rnn_2)

            all_views = concatenate([maxpool, attn, average, word_maxpool, word_average], axis=1)
            x = Dropout(0.5)(all_views)
            dense2 = Dense(self.n_class, activation="softmax")(x)
            res_model = Model(inputs=[char_input, word_input], outputs=dense2)
        else:
            all_views = concatenate([maxpool, attn, average], axis=1)
            #  all_views = attn
            x = Dropout(0.5)(all_views)
            dense2 = Dense(self.n_class, activation="softmax")(x)
            res_model = Model(inputs=[char_input], outputs=dense2)

        #  res_model = Model(inputs=[main_input], outputs=main_output)
        plot_model(res_model, to_file="{}.png".format(self.name), show_shapes=True)
        return res_model
