from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

"""
References:
http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
"""

def make_mlp(hidden_layer_sizes = [], output_dim = None, input_dim = None, alpha = 0.0001, softmax = True):
    architecture = [input_dim] + list(hidden_layer_sizes) + [output_dim]
    # create model
    model = Sequential()
    layers = list(zip(architecture, architecture[1:]))
    
    for i, o in layers[:-1]:
        model.add(Dense(input_dim=i,output_dim=o, init='normal', W_regularizer = l2(alpha)))
        model.add(Activation("relu"))
        
    i, o = layers[-1]
    model.add(Dense(input_dim=i,output_dim=o, init='normal', W_regularizer = l2(alpha)))
    if softmax:
        model.add(Activation("softmax"))

    # Compile model
    if softmax:
        model.compile(loss='kullback_leibler_divergence', optimizer='adam', metrics = ['mse'])
    else:
        model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
    return model



class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None):
        return X.todense()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        return self
    
    def get_params(self, deep=False):
        return {}


def make_lstm(
                max_features = 10000,
                output_dim = 2,
                max_len = 100,
                embedding_size = 64,
                lstm_output_size = 128,
                dropout = 0.25
    ):
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=max_len))
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_output_size, dropout_W=dropout, dropout_U=dropout))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='kullback_leibler_divergence',
                  optimizer='adam',
                  metrics=['mse'])
    return model




def make_conv_lstm(
                max_features = 10000,
                output_dim = 2,
                max_len = 100,
                embedding_size = 64,
                filter_length = 3,
                nb_filter = 64,
                pool_length = 4,
                lstm_output_size = 128,
                dropout = 0.25
    ):
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=max_len))
    model.add(Dropout(dropout))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size, dropout_W=dropout, dropout_U=dropout))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='kullback_leibler_divergence',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



class SequenceTransformer(BaseEstimator, TransformerMixin):
    " Transforms np array of strings into sequences"

    def __init__(self, analyzer='word', max_features=10000, max_len=100):
        self.max_len = max_len
        self.analyzer = analyzer
        self.max_features = max_features
    

    def transform(self, X, y=None):

        try:
            getattr(self, "transformer_")
        except AttributeError:
            raise RuntimeError("You must fit transformer before using it!")

        X_seq = self.transformer_.texts_to_sequences(list(X))
        X_seq = sequence.pad_sequences(X_seq, maxlen=self.max_len)
        return X_seq


    def fit(self, X, y=None):

        if self.analyzer == 'char':
            char_level = True
        elif self.analyzer == 'word':
            char_level = False
        else:
            print("invalid analyzer")
            return

        self.transformer_ = Tokenizer(nb_words=self.max_features, lower=True, char_level = char_level)
        self.transformer_.fit_on_texts(X)

        return self
    


