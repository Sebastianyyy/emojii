import tensorflow as tf
import numpy as np
import pandas as pd


class EmojiModelLSTM(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, max_length, rnn_units, dropout=0.5, vectorize_layer=None):
        super().__init__()
        self.vectorize = vectorize_layer
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=max_length)
        self.hidden1 = tf.keras.layers.LSTM(
            units=128, return_sequences=True)
        self.hidden1_ad = tf.keras.layers.LSTM(
            units=64, return_sequences=False)
        # self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden2 = tf.keras.layers.Dense(64, activation='relu')
        # self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.hidden3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(6, activation='softmax')

    def call(self, inputs):
        vectorize = self.vectorize(inputs)
        embedding = self.embedding(vectorize)
        lstm1 = self.hidden1(embedding)
        lstm2 = self.hidden1_ad(lstm1)
        # drop1 = self.dropout1(lstm2)
        flat1 = self.flatten(lstm2)
        rel1 = self.hidden2(flat1)
        # drop2 = self.dropout2(rel1)
        rel2 = self.hidden3(rel1)
        out = self.output_layer(rel2)

        return out
