import tensorflow as tf
import numpy as np
import pandas as pd
from EmojiModelBi import EmojiiiModelBi
from Preprocessing import Preprocessing
from utils import *
import emoji
from sklearn.model_selection import train_test_split
from EmojiModelRNN import EmojiModelRNN
from EmojiModelLSTM import EmojiModelLSTM
from EmojiModelGRU import EmojiModelGRU
from datasets import load_dataset

if __name__ == "__main__":
    preprocessing = Preprocessing(path=path_set)
    preprocessing.preprocess_data()
    X=preprocessing.get_X()
    y = preprocessing.get_Y()    
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.8)
    text_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=MAX_LEN)
    vectorize_layer.adapt(text_dataset)

    emojiii_model_bi = EmojiiiModelBi(
        max_tokens, embedding_dim, MAX_LEN, rnn_units, dropout,vectorize_layer=vectorize_layer)
    emojiii_model_lstm = EmojiModelLSTM(
        max_tokens, embedding_dim, MAX_LEN, rnn_units, dropout, vectorize_layer=vectorize_layer)
    emojiii_model_rnn = EmojiModelRNN(
        max_tokens, embedding_dim, MAX_LEN, rnn_units, dropout, vectorize_layer=vectorize_layer)
    emojiii_model_gru = EmojiModelGRU(
        max_tokens, embedding_dim, MAX_LEN, rnn_units, dropout, vectorize_layer=vectorize_layer)

    y_train = one_hot(y_train, 6)
    y_val = one_hot(y_val, 6)
    y_test=one_hot(y_test,6)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    loss=tf.keras.losses.CategoricalCrossentropy()
    
    emojiii_model_bi.compile(
        optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    history_bi = emojiii_model_bi.fit(
         X_train, y_train, validation_data=(X_val,y_val), epochs=10, batch_size=64)
      
    emojiii_model_lstm.compile(
        optimizer=optimizer, loss=loss, metrics=['accuracy'])

    history_lstm = emojiii_model_lstm.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)
    
    emojiii_model_gru.compile(
        optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history_gru = emojiii_model_gru.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)
    
    emojiii_model_rnn.compile(
        optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history_rnn = emojiii_model_rnn.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)
    
    plot_accuracy(history_bi, history_lstm, history_gru, history_rnn)
    plot_loss(history_bi,history_lstm,history_gru,history_rnn)
    print("emojiii_model_bi: ", emojiii_model_bi.evaluate(X_test, y_test))
    print("emojii_model_lstm: ", emojiii_model_lstm.evaluate(X_test, y_test))
    print("emoji_model_gru: ", emojiii_model_gru.evaluate(X_test, y_test))
    print("emoji_model_rnn: ", emojiii_model_rnn.evaluate(X_test, y_test))

