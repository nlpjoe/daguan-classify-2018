from model.model_basic import BasicDeepModel
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.core import Dropout
from keras.layers import Conv1D, Dense, Input, Lambda, CuDNNLSTM, CuDNNGRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.utils.vis_utils import plot_model
from model.model_component import AttentionWeightedAverage
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import *

dropout_p = 0.2
hidden_dim_1 = 200
hidden_dim_2 = 100


class RCNNModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=10, config=None):
        name = 'RCNN' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):
        document = Input(shape=(self.word_max_len, ), name='word')
        left_context = Input(shape=(self.word_max_len, ), name='word_left')
        right_context = Input(shape=(self.word_max_len, ), name='word_right')

        doc_mask = Masking(mask_value=self.word_mask_value)(document)
        left_mask = Masking(mask_value=self.word_mask_value)(left_context)
        right_mask = Masking(mask_value=self.word_mask_value)(right_context)

        word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=True, name='word_embedding')
        doc_embed = word_embedding(doc_mask)
        doc_embed = BatchNormalization()(doc_embed)
        doc_embed = Bidirectional(CuDNNGRU(hidden_dim_2, return_sequences=True))(doc_embed)

        l_embed = word_embedding(left_mask)
        r_embed = word_embedding(right_mask)

        l_embed = BatchNormalization()(l_embed)
        r_embed = BatchNormalization()(r_embed)
        forward = Bidirectional(CuDNNGRU(hidden_dim_2, return_sequences=True))(l_embed)

        r_embed = Dropout(dropout_p)(r_embed)
        backward = Bidirectional(CuDNNGRU(hidden_dim_2, return_sequences=True, go_backwards=True))(r_embed)

        # reverse backward
        backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)

        together = concatenate([forward, doc_embed, backward], axis=2)

        x = Conv1D(hidden_dim_2, kernel_size=1, activation='relu')(together)
        # pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)

        # pool_rnn = Dropout(dropout_p)(pool_rnn)

        maxpool = GlobalMaxPooling1D()(x)
        attn = AttentionWeightedAverage()(x)
        average = GlobalAveragePooling1D()(x)
        all_views = concatenate([maxpool, attn, average], axis=1)
        x = Dropout(0.5)(all_views)

        output = Dense(self.n_class, input_dim=hidden_dim_2, activation='softmax')(x)
        model = Model(inputs=[document, left_context, right_context], output=output)
        plot_model(model, to_file="{}.png".format(self.name), show_shapes=True)
        return model

