from keras.layers import *
from keras.models import *
from model.model_component import AttentionWeightedAverage
from model.model_basic import BasicDeepModel
from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.layers.core import Masking


class AttentionModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'attention' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):

        recurrent_units = 60
        dropout_p = 0.2
        #  if self.config.main_feature == 'word':
            #  dropout_p = 0.3
        char_embedding = Embedding(self.max_c_features, self.char_embed_size, weights=[self.char_embedding], trainable=True, name='char_embedding')
        word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=True, name='word_embedding')

        char_input = Input(shape=(self.char_max_len,), name='char')
        word_input = Input(shape=(self.word_max_len,), name='word')

        if self.config.main_feature == 'char':
            embedding = char_embedding
            input = char_input
        else:
            embedding = word_embedding
            input = word_input

        x = Masking(mask_value=self.word_mask_value)(input)
        x = embedding(x)
        x = BatchNormalization()(x)
        #  x = SpatialDropout1D(dropout_p)(x)
      #  x = BatchNormalization()(x)

        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
        x = SpatialDropout1D(dropout_p)(x)
        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
        x = SpatialDropout1D(dropout_p)(x)

        last = Lambda(lambda t: t[:, -1], name='last')(x)
        maxpool = GlobalMaxPooling1D()(x)
        attn = AttentionWeightedAverage()(x)
        average = GlobalAveragePooling1D()(x)

        all_views = concatenate([last, maxpool, attn, average], axis=1)
        x = Dropout(dropout_p)(all_views)
        dense2 = Dense(self.n_class, activation="softmax", kernel_regularizer=regularizers.l2(self.wd))(x)
        res_model = Model(inputs=[input], outputs=dense2)

        #  res_model = Model(inputs=[main_input], outputs=main_output)
        plot_model(res_model, to_file="{}.png".format(self.name), show_shapes=True)
        return res_model
