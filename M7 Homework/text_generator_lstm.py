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
    
    def encode_input_string(self, string):
        v = np.array(list(string)).reshape((self.time_steps, 1))
        return self.encoder.transform(v)
    
    def fit(self, prefix, epochs=2):
        info("Fitting...")
        callbacks = []
        if prefix is not None:
            checkpoint = ModelCheckpoint(prefix + "-{epoch:03d}-{loss:.4f}.hdf5", monitor='loss', verbose=1,
                                         save_best_only=True, mode='min')
            callbacks = [checkpoint]
        self.model.fit(self.X_delayed, self.y_delayed, epochs=epochs, verbose=True, callbacks=callbacks, batch_size=1000)

    def load_weights(self, filename):
        info(f"Loading weights from {filename}...")
        self.model.load_weights(filename)
    
    def predict_from_seed(self, seed, prediction_count):
        info("Predicting output sequence...")
        result = seed
        new_seed = seed
        for i in range(prediction_count):
            inp = self.encode_input_string(new_seed)
            debug(f"{inp=}")
            p = self.encoder.inverse_transform(self.model.predict(np.array([inp])))
            debug(f"{p=}")
            result += p[0][0]
            new_seed = result[-len(seed):]
        return result
    

def main_kafka():
    m = RNNTextModel(open("./kafka_english_the_trial.txt.cleaned").read(), 100)
    m.load_weights("lstm_weights/kafka_trail", 50)
    #m.fit("lstm_weights/kafka_trial", 50)
    #seed = '''done nothing wrong but, one morning, he was arrested. every day at
    #eight in the morning he was brought his breakfast by'''[0:100]
    seed = '''he walked to the bank to find the painter but there was no one
    there. he did not understand who had taken'''[:100]
    print(f"Seed: {seed}")
    print(m.predict_from_seed(seed, 300))

def main_debug():
    m = RNNTextModel("abcbab", 2)
    m.fit(None, 200)
    seed = "ab"
    print(f"Seed: {seed}")
    print(m.predict_from_seed(seed, 15))

def main():
    main_kafka()
    
    