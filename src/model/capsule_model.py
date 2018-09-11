from keras.utils.vis_utils import plot_model
from keras.layers import *
from keras.models import *
from model.model_basic import BasicDeepModel
from model.model_component import Capsule
from keras import regularizers

class CapsuleModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'capsule' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):
        Routings = 5
        Num_capsule = 10
        Dim_capsule = 16
        dropout_p = 0.2

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
        x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
        capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                          share_weights=True)(x)

        if self.config.main_feature == 'all':
            recurrent_units = 60
            word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=False, name='word_embedding')
            word_input = Input(shape=(self.word_max_len,), name='word')
            word_embedding_layer = word_embedding(word_input)
            word_embedding_layer = SpatialDropout1D(0.25)(word_embedding_layer)

            word_rnn_1 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(word_embedding_layer)
            word_rnn_2 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(word_rnn_1)
            word_maxpool = GlobalMaxPooling1D()(word_rnn_2)
            word_average = GlobalAveragePooling1D()(word_rnn_2)

            capsule = Flatten()(capsule)
            capsule = concatenate([capsule, word_maxpool, word_average], axis=-1)
            capsule = Dropout(dropout_p)(capsule)
            dense2 = Dense(self.n_class, activation="softmax")(capsule)
            res_model = Model(inputs=[char_input, word_input], outputs=dense2)
        else:
            capsule = Flatten()(capsule)
            capsule = Dropout(dropout_p)(capsule)
            dense2 = Dense(self.n_class, activation="softmax", kernel_regularizer=regularizers.l2(self.wd))(capsule)
            res_model = Model(inputs=[input], outputs=dense2)

        plot_model(res_model, to_file="{}.png".format(self.name), show_shapes=True)
        return res_model
