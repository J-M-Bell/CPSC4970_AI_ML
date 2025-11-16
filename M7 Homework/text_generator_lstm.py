from keras import Input, activations
from keras.callbacks import ModelCheckpoint
from keras.layers import SimpleRNN, Dense, LSTM, Dropout
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
import numpy as np

PRINT_DEBUG = False
PRINT_INFO = True


def debug(*args):
    if PRINT_DEBUG:
        print(*args)


def info(*args):
    if PRINT_INFO:
        print(*args)


def time_delayed(seq, delay):
    features = []
    targets = []
    for target_index in range(delay, len(seq)):
        features.append(seq[target_index - delay:target_index])
        targets.append(seq[target_index])
    return np.array(features), np.array(targets)

def encode_sequence(sequence):
    """Given a string, I encode it letter by letter (each letter is a sample). I return the
    result and the encoder."""
    info("Encoding inputs...")
    debug(f"{sequence}")
    encoder = OneHotEncoder(sparse=False)
    result = encoder.fit_transform(np.reshape(sequence, (len(sequence), 1)))
    info("Number of input characters:", len(encoder.categories_[0]))
    debug("Input categories:", encoder.categories_[0])
    info(f"{result.shape=}")
    debug(result)
    return result, encoder

class RNNTextModel:
    def __init__(self, training_string, delay_length=100):
        info("Number of distinct characters:", len(set(training_string)))
        debug("Distinct characters:", set(training_string))
        self.time_steps = delay_length
        encoded_training_data, self.encoder = encode_sequence(list(training_string))
        debug("encoded_training_data:", encoded_training_data)
        self.X_delayed, self.y_delayed = time_delayed(encoded_training_data, self.time_steps)
        self.model = self.create_model(self.X_delayed.shape, self.y_delayed.shape)

    def create_model(input_shape, output_shape):
        info("Creating model...")
        info("Input shape:", input_shape[1:])
        model = Sequential(
            [Input(shape=input_shape[1:]),
            LSTM(256, return_sequences=True, activation=activations.tanh),
            Dropout(0.2),
            LSTM(256, activation=activations.tanh),
            Dropout(0.2),
            Dense(output_shape[1], activation=activations.softmax)]
        )
        model.summary()
        model.compile(optimizer="adam", loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        return model
    
    def 