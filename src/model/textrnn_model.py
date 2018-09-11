from model.model_basic import BasicDeepModel
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.core import Dropout
from keras.layers import Conv1D, Dense, Input, Lambda, CuDNNLSTM, CuDNNGRU, LSTM
from keras.models import Model
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Masking
from model.model_component import AttentionWeightedAverage

dropout_p = 0.1
hidden_dim = 128


class TextRNNModel(BasicDeepModel):
    def __init__(self, name='basicModel', num_flods=5, config=None):
        name = 'TextRnn' + config.main_feature
        BasicDeepModel.__init__(self, name=name, n_folds=num_flods, config=config)

    def create_model(self):
        word = Input(shape=(self.word_max_len, ), name='word')

        w_mask = Masking(mask_value=self.word_mask_value)(word)
        word_embedding = Embedding(self.max_w_features, self.word_embed_size, weights=[self.word_embedding], trainable=True, name='word_embedding')
        word_embed = word_embedding(w_mask)
        l_lstm = Bidirectional(CuDNNLSTM(hidden_dim, return_sequences=True))(word_embed)

        l_lstm = AttentionWeightedAverage()(l_lstm)
        output = Dense(self.n_class, activation='softmax')(l_lstm)
        model = Model(inputs=[word], output=output)
        plot_model(model, to_file="{}.png".format(self.name), show_shapes=True)
        return model
