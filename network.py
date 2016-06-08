# Convolutional neural network 
import sys

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.regularizers import l2
from keras.utils.visualize_util import plot

from utils import data

# Train / Test config
BATCH_SIZE = 1
EPOCHS = 1000

# Model config
LEARNING_RATE = 0.01
MOMENTUM = 1e-5
DECAY = 0 #.005

# Objective config
LOSS_FUNCTION = 'mse'

# L2 Regularization
WEIGHT_DECAY = 0.0005


# Define custom callbacks
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def build_model():
    # build a model to train
    model = Sequential()

    # LAYER 01
    model.add(
        Convolution2D(
            128, # num of filters
            19, # kernel size
            19, # kernel size
            subsample=(1, 1), # stride length
            border_mode='valid',
            init='glorot_uniform', # same as caffe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
            input_shape=(3, data.IN_IMG_HEIGHT, data.IN_IMG_WIDTH)
        )
    )
    model.add(Activation('relu'))

    # LAYER 02
    model.add(
        Convolution2D(
            320, # num of filters
            1, # nb_row
            1, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 03
    model.add(
        Convolution2D(
            320, # num of filters
            1, # nb_row
            1, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 04
    model.add(
        Convolution2D(
            320, # num of filters
            1, # nb_row
            1, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 05
    model.add(
        Convolution2D(
            128, # num of filters
            1, # nb_row
            1, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 06
    model.add(
        Convolution2D(
            128, # num of filters
            3, # nb_row
            3, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 07
    model.add(
        Convolution2D(
            512, # num of filters
            1, # nb_row
            1, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 08
    # NOTE: group=4 ignored here
    # https://github.com/BVLC/caffe/issues/778
    model.add(
        Convolution2D(
            128, # num of filters
            5, # nb_row
            5, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 09
    # NOTE: group=2 ignored here
    model.add(
        Convolution2D(
            128, # num of filters
            1, # nb_row
            1, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 10
    model.add(
        Convolution2D(
            128, # num of filters
            3, # nb_row
            3, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 11
    # NOTE: group=2 ignored here
    model.add(
        Convolution2D(
            128, # num of filters
            5, # nb_row
            5, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 12
    # NOTE: group=2 ignored here
    model.add(
        Convolution2D(
            128, # num of filters
            5, # nb_row
            5, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 13
    # NOTE: group=2 ignored here
    model.add(
        Convolution2D(
            256, # num of filters
            5, # nb_row
            5, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 14
    # NOTE: group=2 ignored here
    model.add(
        Convolution2D(
            64, # num of filters
            7, # nb_row
            7, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )
    model.add(Activation('relu'))

    # LAYER 15
    model.add(
        Convolution2D(
            3, # num of filters
            7, # nb_row
            7, # nb_col
            subsample=(1, 1), # stre length
            border_mode='valid',
            init='glorot_uniform', # same as cfe's xavier
            W_regularizer=l2(WEIGHT_DECAY),
            # b_regularizer=None,
            dim_ordering='th', # (batch, *depth*, width, height)
        )
    )

    # configure the optimizer
    sgd = SGD(
        lr=LEARNING_RATE,
        decay=DECAY,
        momentum=MOMENTUM,
        nesterov=True
    )

    # compile the model
    model.compile(
        loss=LOSS_FUNCTION,
        optimizer=sgd,
        # metrics=['accuracy']
    )

    return model

def train_model(model, train, test=None):
    history = LossHistory()
    # train the model
    if test:
        hist = model.fit(
            train[0],
            train[1],
            batch_size=BATCH_SIZE,
            nb_epoch=EPOCHS,
            verbose=1,
            validation_data=(test[0], test[1]),
            callbacks=[
                keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    verbose=0,
                    mode='min'),
                history
            ]
        )
    else:
        hist = model.fit(
            train[0],
            train[1],
            batch_size=BATCH_SIZE,
            nb_epoch=EPOCHS,
            verbose=1,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    verbose=0,
                    mode='min'),
                history
            ]
        )

    return hist, history

def evaluate_model(model, test):
    return model.evaluate(test[0], test[1], verbose=1)

def predict_model(model, sample):
    return model.predict(sample, batch_size=1)

def load_model(name):
    with open('{}.json'.format(name), 'r') as json_file:
        # load the model architecture
        model = model_from_json(json_file.read())

        # load the model weights
        model.load_weights('{}.h5'.format(name))

        print 'WARNING: Before it can be used, it should be compiled.'
        
        return model

def save_model(model, name):
    # save the model weights
    model.save_weights('{}.h5'.format(name), overwrite=True)

    # save the model architecture
    with open('{}.json'.format(name), 'w') as json_file:
        json_file.write(model.to_json())


def vizualize_model(model, filename):
    model.summary()
    model.get_config()
    plot(model, to_file='{}.png'.format(filename))
    
