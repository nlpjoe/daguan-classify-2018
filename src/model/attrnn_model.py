from model.model_basic import BasicDeepModel
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.core import Dropout
from keras.layers import Conv1D, Dense, Input, Lambda, CuDNNLSTM, CuDNNGRU, LSTM
from keras.models import Model
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.layers.wrappers import Bidirectional
from model.model_component import AttentionWeightedAverage
from keras.layers.normalization import BatchNormalization

dropout_p = 0.2
hidden_dim = 100


class AttentionRNNModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'ATTRnn' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):
        word = Input(shape=(self.word_max_len, ), name='word')

        word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=True, name='word_embedding')
        x = word_embedding(word)
        x = BatchNormalization()(x)

        x = Dropout(dropout_p)(x)
        x = Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True, unit_forget_bias=False))(x)
        x = Dropout(dropout_p)(x)
        x = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))(x)
        x = Dropout(dropout_p)(x)
        x = AttentionWeightedAverage()(x)
        x = Dropout(dropout_p)(x)
        output = Dense(self.n_class, activation='softmax')(x)
        model = Model(inputs=[word], output=output)
        plot_model(model, to_file="{}.png".format(self.name), show_shapes=True)
        return model

