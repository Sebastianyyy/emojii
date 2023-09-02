import emoji
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import Preprocessing
import pandas as pd
embedding_dim = 300
MAX_LEN = 70
rnn_units = 128
dropout = 0.4
path_set = 'dair-ai/emotion'
max_tokens = 4000
def label_to_emoji(target):
    if target == 0:
        return emoji.emojize(":sad_but_relieved_face:")
    elif target == 1:
        return emoji.emojize(":face_with_tears_of_joy:")
    elif target == 2:
        return emoji.emojize(":heart_exclamation:")
    elif target == 3:
        return emoji.emojize(":angry_face::")
    elif target == 4:
        return emoji.emojize(":fearful_face:")
    elif target == 5:
        return emoji.emojize(":hushed_face:")

def one_hot(y, number_of_labels):
    return tf.one_hot(y, number_of_labels)


# def read_embedding(path):
#     embedding_index = {}
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             word, coef = line.split(maxsplit=1)
#             coef = np.fromstring(coef, 'f', sep=' ')
#             embedding_index[word] = coef
#     return embedding_index


# def get_embedding_matrix(word_index, embedding_index, embedding_dim=100):
#     embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
#     for word, i in word_index.items():
#         embedding_vector = embedding_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#         else:
#             print(word)
#     return embedding_matrix


def plot_accuracy(history_bi, history_lstm, history_gru, history_rnn):
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0,0].plot(history_bi.history['accuracy'],label='train')
    ax[0,0].plot(history_bi.history['val_accuracy'],label='val')
    ax[0,0].set_title('BiRNN')
    ax[0,0].legend()
    ax[0, 1].plot(history_lstm.history['accuracy'], label='train')
    ax[0, 1].plot(history_lstm.history['val_accuracy'], label='val')
    ax[0,1].set_title('LSTMRNN')
    ax[0,1].legend()
    ax[1, 0].plot(history_gru.history['accuracy'], label='train')
    ax[1, 0].plot(history_gru.history['val_accuracy'], label='val')
    ax[1,0].set_title('GRURNN')
    ax[1,0].legend()
    ax[1, 1].plot(history_rnn.history['accuracy'], label='train')
    ax[1, 1].plot(history_rnn.history['val_accuracy'], label='val')
    ax[1,1].set_title('RNN')
    ax[1,1].legend()
    plt.savefig('accuracy.png')
    
def plot_loss(history_bi, history_lstm, history_gru, history_rnn):
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0, 0].plot(history_bi.history['loss'], label='train')
    ax[0, 0].plot(history_bi.history['val_loss'], label='val')
    ax[0, 0].set_title('BiRNN')
    ax[0, 0].legend()
    ax[0, 1].plot(history_lstm.history['loss'], label='train')
    ax[0, 1].plot(history_lstm.history['val_loss'], label='val')
    ax[0, 1].set_title('LSTMRNN')
    ax[0, 1].legend()
    ax[1, 0].plot(history_gru.history['loss'], label='train')
    ax[1, 0].plot(history_gru.history['val_loss'], label='val')
    ax[1, 0].set_title('GRURNN')
    ax[1, 0].legend()
    ax[1, 1].plot(history_rnn.history['loss'], label='train')
    ax[1, 1].plot(history_rnn.history['val_loss'], label='val')
    ax[1, 1].set_title('RNN')
    ax[1, 1].legend()
    plt.savefig('loss.png')
    
    
def test(text,model,preprocessing):
    text = transform(text, preprocessing)
    print(text)
    return label_to_emoji(np.argmax(model.predict(text)))

def transform(text,preprocessing):
    text = (pd.Series([text]))
    text = preprocessing.remove_stop_words(text)
    text = preprocessing.steming_and_tokenization(text)
    return text
