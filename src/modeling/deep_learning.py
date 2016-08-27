from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

def make_MLP(hidden_layer_sizes = [], output_dim = None, input_dim = None, alpha = 0.0001, softmax = True):
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
        model.compile(loss='kullback_leibler_divergence', optimizer='adam', metrics = ['accuracy'])
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


def make_conv_rnn(
                max_features = 10000,
                output_dim = 2,
                maxlen = 100,
                embedding_size = 128,
                filter_length = 3,
                nb_filter = 64,
                pool_length = 2,
                lstm_output_size = 64,
                dropout = 0.25
    ):
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(dropout))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='kullback_leibler_divergence',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

class SequenceTransformer(TransformerMixin):

    def __init__(self, maxlen=100):
        self.maxlen = maxlen

    def transform(self, X, y=None):
        X_seq = []
        for i in range(X.shape[0]):
            X_seq.append(np.nonzero(X[i, :])[1])
        X_seq = np.array(X_seq)
        X_seq = sequence.pad_sequences(X_seq, maxlen=self.maxlen)
        return X_seq

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        return self
    
    def get_params(self, deep=False):
        return {}



def apply_transform(data, old_xtype, transformer, new_xtype,  nss = ['user', 'article'], samples = ['random', 'blocked'], splits = ['train', 'dev','test']):
    for ns in nss:
        for sample in samples:
            for split in splits:
                old = data[ns][sample][split]['x'][old_xtype]
                new = transformer.transform(old)
                new = pd.DataFrame(data = new, index = old.index)
                data[ns][sample][split]['x'][new_xtype] = new
    return data

