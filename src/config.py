from model.lightgbm_model import LightGbmModel
# from model.xgboost_model import XgboostModel
from model.textcnn_model import TextCNNModel
from model.lstmgru_model import LstmgruModel
from model.attention import AttentionModel
from model.attention1 import AttentionNoLamdaModel
from model.convlstm_model import ConvlstmModel
from model.dpcnn_model import DpcnnModel
from model.capsule_model import CapsuleModel
from model.rcnn_model import RCNNModel
from model.textrnn_model import TextRNNModel
from model.attrnn_model import AttentionRNNModel
from model.simplecnn_model import SimpleCNNModel

class Config(object):

    """Docstring for Config. """

    def __init__(self):
        """TODO: to be defined1. """
        self.model = {
            # 'xgboost': XgboostModel,
            'lightgbm': LightGbmModel,

            # dl model
            'textcnn': TextCNNModel,
            'textrnn': TextRNNModel,
            'lstmgru': LstmgruModel,
            'attention': AttentionModel,
            'attention1': AttentionNoLamdaModel,
            'rnn_attention': AttentionRNNModel,
            'convlstm': ConvlstmModel,
            'dpcnn': DpcnnModel,
            'rcnn': RCNNModel,
            'capsule': CapsuleModel,
            'simplecnn': SimpleCNNModel,
        }
        self.n_class = 19
        # 9-5
        # self.WORD_MAXLEN = 1000
        self.CHAR_MAXLEN = 2000
        self.WORD_MAXLEN = self.CHAR_MAXLEN

        self.EMBED_SIZE = 100
        self.BATCH_SIZE = 128
        self.main_feature = 'word'
        self.wd = 1e-6
        self.date = '0908'

        #  self.article_w2v_file = '../data/word2vec-models/word2vec.char.{}d.model'.format(self.EMBED_SIZE)
        #  self.word_seg_w2v_file = '../data/word2vec-models/word2vec.word.{}d.model'.format(self.EMBED_SIZE)
        self.article_w2v_file = '../data/word2vec-models/word2vec.char.{}d.mfreq3.model'.format(self.EMBED_SIZE)
        self.word_seg_w2v_file = '../data/word2vec-models/word2vec.word.{}d.mfreq3.model'.format(self.EMBED_SIZE)
        self.chi_word_file = '../data/chi_words_40000.pkl'
        self.chi_char_file = '../data/chi_char_3000.pkl'

        # self.TRAIN_X = '../data/Clean.train.csv'
        self.TRAIN_X = '../data/New.train.csv'
        # self.TRAIN_X = '../data/Enhance.train.csv'
        self.TEST_X = '../data/New.test.csv'

